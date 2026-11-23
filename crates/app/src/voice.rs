//! Speaking into a terminal.
//!
//! Ctrl+Shift+R starts recording for the focused terminal, and pressing it
//! again, or Ctrl+Q, stops, from whichever terminal has focus by then;
//! whisper.cpp turns the take into text on this machine, and the text is
//! typed at the prompt of the terminal it was recorded for, the way a paste
//! types it. It
//! is not sent: a misheard word is fixed before Enter, not after. The text
//! also goes to the clipboard, so a take whose terminal closed meanwhile is
//! not lost.
//!
//! The microphone is open only while recording. A stream kept open for the
//! whole run would have macOS show its recording light the whole time.
//! cpal's stream is not `Send` on every platform, so it stays here, on the
//! UI thread; its callback only appends samples under a lock.
//!
//! Transcription runs on one thread of its own, started with the first take
//! and kept for the run, so the model (half a gigabyte, several seconds to
//! load and to compile its GPU kernels the first time) is read once rather
//! than per take — and that once is while the first take is being spoken,
//! not after it: measured, the first take took 7.4 s to come back otherwise. Takes queue there: the next
//! recording can start while the last one is still being transcribed, and
//! each take's text goes to the terminal it was recorded for.
//!
//! Whisper wants 16 kHz mono; a Mac microphone gives 48 kHz, sometimes in
//! stereo. [`to_whisper_rate`] mixes and resamples it.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, Sender};
use eframe::egui::{self, Color32, FontId};
use giverny_core::config::VoiceConfig;
use giverny_core::tabs::TabId;

use crate::chrome::Chrome;

/// The rate whisper.cpp takes its audio at.
const WHISPER_RATE: u32 = 16_000;

/// The model looked for when `voice.model` is empty, in Giverny's directory.
pub const DEFAULT_MODEL: &str = "whisper/ggml-large-v3-turbo-q5_0.bin";

/// Shorter than this is a key pressed twice, not something said.
const TOO_SHORT: Duration = Duration::from_millis(400);

/// A take is cut here: whisper.cpp would manage longer, but nobody means to
/// dictate this long, and a recording left running would grow forever.
const LONGEST: Duration = Duration::from_secs(5 * 60);

/// How long a notice ("no model", "nothing heard") stays on the terminal.
const NOTICE_FOR: Duration = Duration::from_secs(6);

/// Words heard, and the terminal they were recorded for.
pub struct Heard {
    pub tab: TabId,
    pub text: String,
}

/// What a terminal shows of voice input, if anything.
#[derive(Debug, Clone, PartialEq)]
pub enum Badge {
    Recording(Duration),
    Transcribing,
    Notice(String),
}

struct Take {
    tab: TabId,
    started: Instant,
    samples: Arc<Mutex<Vec<f32>>>,
    rate: u32,
    channels: u16,
    // Dropping it closes the microphone.
    _stream: cpal::Stream,
}

struct Job {
    tab: TabId,
    audio: Vec<f32>,
    language: String,
    model: PathBuf,
}

enum Work {
    /// Have the model ready: recording has started.
    Load(PathBuf),
    Take(Job),
}

/// The worker's answer to a job: the words, or why there are none.
struct Done {
    tab: TabId,
    result: Result<String, String>,
}

pub struct Voice {
    take: Option<Take>,
    /// Takes sent to the worker and not yet answered, by terminal.
    queued: Vec<TabId>,
    jobs: Option<Sender<Work>>,
    done_tx: Sender<Done>,
    done_rx: Receiver<Done>,
    notice: Option<(TabId, String, Instant)>,
}

impl Voice {
    pub fn new() -> Self {
        let (done_tx, done_rx) = crossbeam_channel::unbounded();
        Self {
            take: None,
            queued: Vec::new(),
            jobs: None,
            done_tx,
            done_rx,
            notice: None,
        }
    }

    /// Ctrl+Shift+R in `tab`: start recording for it, or stop the take in
    /// progress, whichever terminal that was recorded for.
    pub fn toggle(&mut self, tab: TabId, cfg: &VoiceConfig, base: &Path) {
        if !cfg.enabled {
            return;
        }
        if self.take.is_some() {
            self.stop(cfg, base);
            return;
        }
        let model = model_path(cfg, base);
        // Found out now rather than after a minute of talking.
        if !model.is_file() {
            self.tell(
                tab,
                format!(
                    "no whisper model at {}",
                    giverny_core::short_path(&model, 60)
                ),
            );
            return;
        }
        match open_microphone() {
            Ok((stream, samples, rate, channels)) => {
                self.notice = None;
                let _ = self.worker().send(Work::Load(model));
                self.take = Some(Take {
                    tab,
                    started: Instant::now(),
                    samples,
                    rate,
                    channels,
                    _stream: stream,
                });
            }
            Err(e) => self.tell(tab, format!("microphone: {e}")),
        }
    }

    /// Whether a take is being recorded, in any terminal.
    pub fn recording(&self) -> bool {
        self.take.is_some()
    }

    /// Stop the take in progress and send it to be transcribed.
    pub fn stop(&mut self, cfg: &VoiceConfig, base: &Path) {
        let Some(take) = self.take.take() else {
            return;
        };
        let tab = take.tab;
        let held = take.started.elapsed();
        let raw = std::mem::take(&mut *take.samples.lock().unwrap_or_else(|e| e.into_inner()));
        let (rate, channels) = (take.rate, take.channels);
        drop(take);
        if held < TOO_SHORT || raw.is_empty() {
            self.tell(tab, "too short to hear anything".into());
            return;
        }
        let job = Job {
            tab,
            audio: to_whisper_rate(&raw, channels, rate),
            language: cfg.language.trim().to_lowercase(),
            model: model_path(cfg, base),
        };
        if self.worker().send(Work::Take(job)).is_ok() {
            self.queued.push(tab);
        } else {
            self.tell(tab, "transcriber stopped; restart Giverny".into());
        }
    }

    fn worker(&mut self) -> &Sender<Work> {
        self.jobs
            .get_or_insert_with(|| spawn_worker(self.done_tx.clone()))
    }

    /// Words heard since the last frame, to be typed; and the take cut at
    /// [`LONGEST`], if one ran that long.
    pub fn poll(&mut self, cfg: &VoiceConfig, base: &Path) -> Vec<Heard> {
        if self
            .take
            .as_ref()
            .is_some_and(|t| t.started.elapsed() >= LONGEST)
        {
            self.stop(cfg, base);
        }
        let mut heard = Vec::new();
        while let Ok(done) = self.done_rx.try_recv() {
            if let Some(i) = self.queued.iter().position(|&t| t == done.tab) {
                self.queued.remove(i);
            }
            match done.result {
                Ok(text) if !text.is_empty() => heard.push(Heard {
                    tab: done.tab,
                    text,
                }),
                Ok(_) => self.tell(done.tab, "nothing heard".into()),
                Err(e) => self.tell(done.tab, e),
            }
        }
        if self
            .notice
            .as_ref()
            .is_some_and(|(_, _, at)| at.elapsed() >= NOTICE_FOR)
        {
            self.notice = None;
        }
        heard
    }

    /// Something is moving that the screen should keep up with: the
    /// recording clock, a transcription to collect, a notice to take down.
    pub fn busy(&self) -> bool {
        self.take.is_some() || !self.queued.is_empty() || self.notice.is_some()
    }

    pub fn badge(&self, tab: TabId) -> Option<Badge> {
        if let Some(take) = self.take.as_ref().filter(|t| t.tab == tab) {
            return Some(Badge::Recording(take.started.elapsed()));
        }
        if self.queued.contains(&tab) {
            return Some(Badge::Transcribing);
        }
        self.notice
            .as_ref()
            .filter(|(t, _, _)| *t == tab)
            .map(|(_, text, _)| Badge::Notice(text.clone()))
    }

    fn tell(&mut self, tab: TabId, text: String) {
        tracing::info!("voice: {text}");
        self.notice = Some((tab, text, Instant::now()));
    }
}

/// The model file `cfg` names, relative paths taken from Giverny's
/// directory; [`DEFAULT_MODEL`] there when it names none.
pub fn model_path(cfg: &VoiceConfig, base: &Path) -> PathBuf {
    let named = cfg.model.trim();
    if named.is_empty() {
        return base.join(DEFAULT_MODEL);
    }
    if let Some(rest) = named.strip_prefix("~/")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(rest);
    }
    base.join(named)
}

type Opened = (cpal::Stream, Arc<Mutex<Vec<f32>>>, u32, u16);

fn open_microphone() -> Result<Opened, String> {
    let device = cpal::default_host()
        .default_input_device()
        .ok_or("no input device")?;
    let config = device.default_input_config().map_err(|e| e.to_string())?;
    let rate = config.sample_rate();
    let channels = config.channels();
    let samples = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&samples);
    let on_error = |e: cpal::Error| tracing::warn!("voice: microphone: {e}");
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            config.into(),
            move |data: &[f32], _: &_| append(&sink, data.iter().copied()),
            on_error,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_input_stream(
            config.into(),
            move |data: &[i16], _: &_| append(&sink, data.iter().map(|&s| f32::from(s) / 32_768.0)),
            on_error,
            None,
        ),
        cpal::SampleFormat::I32 => device.build_input_stream(
            config.into(),
            move |data: &[i32], _: &_| {
                append(&sink, data.iter().map(|&s| s as f32 / 2_147_483_648.0))
            },
            on_error,
            None,
        ),
        other => return Err(format!("unsupported sample format {other}")),
    }
    .map_err(|e| e.to_string())?;
    stream.play().map_err(|e| e.to_string())?;
    Ok((stream, samples, rate, channels))
}

fn append(sink: &Mutex<Vec<f32>>, samples: impl Iterator<Item = f32>) {
    sink.lock()
        .unwrap_or_else(|e| e.into_inner())
        .extend(samples);
}

/// Interleaved samples at `rate` to 16 kHz mono: the channels averaged, then
/// each output sample the mean of the input it spans. A plain mean is a poor
/// low-pass, but speech sits well under the 8 kHz it has to keep, and Whisper
/// was trained on worse.
pub fn to_whisper_rate(interleaved: &[f32], channels: u16, rate: u32) -> Vec<f32> {
    let channels = usize::from(channels.max(1));
    let mono: Vec<f32> = interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect();
    if rate == WHISPER_RATE || rate == 0 {
        return mono;
    }
    let step = f64::from(rate) / f64::from(WHISPER_RATE);
    let len = (mono.len() as f64 / step).floor() as usize;
    (0..len)
        .map(|i| {
            let from = (i as f64 * step) as usize;
            let to = (((i + 1) as f64 * step) as usize).clamp(from + 1, mono.len());
            mono[from..to].iter().sum::<f32>() / (to - from) as f32
        })
        .collect()
}

/// whisper.cpp's text, tidied: its segments joined, and what it writes for
/// silence or noise rather than speech — `[BLANK_AUDIO]`, `[Music]`,
/// `(silence)` — dropped when that is all a segment holds.
pub fn tidy<'a>(segments: impl IntoIterator<Item = &'a str>) -> String {
    let words: Vec<&str> = segments
        .into_iter()
        .map(str::trim)
        .filter(|s| {
            let marker = (s.starts_with('[') && s.ends_with(']'))
                || (s.starts_with('(') && s.ends_with(')'));
            !s.is_empty() && !marker
        })
        .collect();
    words.join(" ")
}

fn spawn_worker(done: Sender<Done>) -> Sender<Work> {
    let (tx, rx) = crossbeam_channel::unbounded::<Work>();
    let spawned = std::thread::Builder::new()
        .name("voice".into())
        .spawn(move || {
            // whisper.cpp logs every layer it loads to stderr; through
            // tracing it goes where the rest of Giverny's log goes.
            whisper_rs::install_logging_hooks();
            let mut loaded: Option<(PathBuf, whisper_rs::WhisperContext)> = None;
            for work in rx {
                let job = match work {
                    Work::Load(model) => {
                        warm_up(&mut loaded, model);
                        continue;
                    }
                    Work::Take(job) => job,
                };
                let result = transcribe(&mut loaded, &job);
                if done
                    .send(Done {
                        tab: job.tab,
                        result,
                    })
                    .is_err()
                {
                    break;
                }
            }
        });
    if let Err(e) = spawned {
        tracing::warn!("voice: no worker thread: {e}");
    }
    tx
}

type Loaded = Option<(PathBuf, whisper_rs::WhisperContext)>;

/// Load `model` and run it once over a second of silence. Loading alone
/// leaves the GPU kernels to be compiled by the first real take. Failures
/// are not reported here: the take that follows meets them again, and says.
fn warm_up(loaded: &mut Loaded, model: PathBuf) {
    if loaded.as_ref().is_some_and(|(path, _)| *path == model) {
        return;
    }
    let silence = Job {
        tab: TabId(0),
        audio: vec![0.0; WHISPER_RATE as usize],
        language: "en".into(),
        model,
    };
    let _ = transcribe(loaded, &silence);
}

fn load<'a>(
    loaded: &'a mut Loaded,
    model: &Path,
) -> Result<&'a whisper_rs::WhisperContext, String> {
    // The model is kept until the setting names another.
    if loaded.as_ref().is_none_or(|(path, _)| path != model) {
        *loaded = None;
        let path = model.to_str().ok_or("model path is not UTF-8")?;
        let ctx = whisper_rs::WhisperContext::new_with_params(
            path,
            whisper_rs::WhisperContextParameters::default(),
        )
        .map_err(|e| format!("loading the model: {e}"))?;
        *loaded = Some((model.to_path_buf(), ctx));
    }
    loaded
        .as_ref()
        .map(|(_, ctx)| ctx)
        .ok_or_else(|| "no model".into())
}

fn transcribe(loaded: &mut Loaded, job: &Job) -> Result<String, String> {
    let ctx = load(loaded, &job.model)?;
    let mut state = ctx.create_state().map_err(|e| e.to_string())?;
    let mut params =
        whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });
    let language = if job.language.is_empty() {
        "auto"
    } else {
        job.language.as_str()
    };
    params.set_language(Some(language));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_no_timestamps(true);
    // Each take stands alone: text carried over from the last one only
    // ever turns up as a repeat in this one.
    params.set_no_context(true);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(8));
    params.set_n_threads(threads as i32);
    state
        .full(params, &job.audio)
        .map_err(|e| format!("transcribing: {e}"))?;
    let segments: Vec<String> = state
        .as_iter()
        .filter_map(|s| s.to_str_lossy().ok().map(|t| t.into_owned()))
        .collect();
    Ok(tidy(segments.iter().map(String::as_str)))
}

/// The badge in a terminal's corner: the recording clock, "transcribing",
/// or a notice.
pub fn paint_badge(painter: &egui::Painter, grid: egui::Rect, badge: &Badge, c: &Chrome) {
    let (dot, text) = match badge {
        Badge::Recording(held) => {
            let secs = held.as_secs();
            (
                Some(c.poppy),
                format!("rec {}:{:02} · Ctrl+Q stops", secs / 60, secs % 60),
            )
        }
        Badge::Transcribing => (Some(c.amber), "transcribing…".into()),
        Badge::Notice(text) => (None, format!("voice: {text}")),
    };
    let font = FontId::monospace(11.5);
    let galley = painter.layout_no_wrap(text, font, c.fg);
    let pad = egui::vec2(8.0, 4.0);
    let dot_w = if dot.is_some() { 12.0 } else { 0.0 };
    let size = galley.size() + pad * 2.0 + egui::vec2(dot_w, 0.0);
    let rect = egui::Rect::from_min_size(grid.right_bottom() - size - egui::vec2(10.0, 10.0), size);
    painter.rect_filled(rect, 4.0, c.panel);
    painter.rect_stroke(
        rect,
        4.0,
        egui::Stroke::new(1.0, dot.unwrap_or(c.dim)),
        egui::StrokeKind::Inside,
    );
    if let Some(dot) = dot {
        painter.circle_filled(
            egui::pos2(rect.left() + pad.x + 4.0, rect.center().y),
            4.0,
            dot,
        );
    }
    painter.galley(
        egui::pos2(rect.left() + pad.x + dot_w, rect.top() + pad.y),
        galley,
        Color32::PLACEHOLDER,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixes_stereo_and_keeps_whisper_rate() {
        let stereo = [0.2, 0.4, -1.0, 1.0, 0.5, 0.5];
        assert_eq!(
            to_whisper_rate(&stereo, 2, WHISPER_RATE),
            vec![0.3, 0.0, 0.5]
        );
    }

    #[test]
    fn resamples_48k_to_16k_by_thirds() {
        let mono: Vec<f32> = (0..48_000).map(|i| (i % 3) as f32).collect();
        let out = to_whisper_rate(&mono, 1, 48_000);
        assert_eq!(out.len(), 16_000);
        assert!(out.iter().all(|&s| (s - 1.0).abs() < 1e-6));
    }

    #[test]
    fn resamples_44_1k_to_the_right_length() {
        let out = to_whisper_rate(&vec![0.5; 44_100], 1, 44_100);
        assert_eq!(out.len(), 16_000);
        assert!(out.iter().all(|&s| (s - 0.5).abs() < 1e-6));
    }

    #[test]
    fn tidy_drops_non_speech_markers() {
        assert_eq!(
            tidy([" Hello there.", " [BLANK_AUDIO]", " How are you?"]),
            "Hello there. How are you?"
        );
        assert_eq!(tidy(["(silence)", "[Music]", "  "]), "");
        assert_eq!(tidy([" שלום, מה שלומך?"]), "שלום, מה שלומך?");
    }

    #[test]
    fn model_path_defaults_into_the_base() {
        let base = Path::new("/base");
        let cfg = VoiceConfig::default();
        assert_eq!(model_path(&cfg, base), base.join(DEFAULT_MODEL));
        let cfg = VoiceConfig {
            model: "/models/ggml-small.bin".into(),
            ..VoiceConfig::default()
        };
        assert_eq!(
            model_path(&cfg, base),
            PathBuf::from("/models/ggml-small.bin")
        );
    }

    #[test]
    fn a_disabled_toggle_records_nothing() {
        let mut voice = Voice::new();
        let cfg = VoiceConfig {
            enabled: false,
            ..VoiceConfig::default()
        };
        voice.toggle(TabId(1), &cfg, Path::new("/nonexistent"));
        assert!(!voice.busy());
    }

    #[test]
    fn a_missing_model_is_told_without_opening_the_microphone() {
        let mut voice = Voice::new();
        voice.toggle(TabId(1), &VoiceConfig::default(), Path::new("/nonexistent"));
        assert!(
            matches!(voice.badge(TabId(1)), Some(Badge::Notice(t)) if t.starts_with("no whisper model"))
        );
        assert_eq!(voice.badge(TabId(2)), None);
    }
}
