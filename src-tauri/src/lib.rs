use std::env;
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;
use tauri::RunEvent;

static BACKEND_PROCESS: Mutex<Option<Child>> = Mutex::new(None);

fn packaged_root() -> PathBuf {
    let executable_dir = env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(PathBuf::from));
    let source_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source_root = source_dir.parent().unwrap_or(&source_dir).to_path_buf();

    // A portable build keeps backend/ beside the executable. During `tauri dev`
    // the executable lives under target/, so fall back to the source checkout.
    executable_dir
        .filter(|dir| dir.join("backend").is_dir())
        .unwrap_or(source_root)
}

fn local_python(root: &PathBuf) -> Option<PathBuf> {
    // pythonw keeps the backend console hidden in a packaged Windows build.
    // The regular python executable is retained as a fallback for diagnostics.
    for name in ["pythonw.exe", "python.exe"] {
        let candidate = root.join("python").join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn ensure_backend_running() {
    let addr = "127.0.0.1:48653";
    if TcpStream::connect_timeout(&addr.parse().unwrap(), Duration::from_millis(250)).is_ok() {
        println!("[Tauri] Backend is already running on {}.", addr);
        return;
    }

    let root = packaged_root();
    let py_bin = local_python(&root).or_else(|| {
        for name in ["pythonw", "python"] {
            if Command::new(name).arg("--version").output().is_ok() {
                return Some(PathBuf::from(name));
            }
        }
        None
    });

    let Some(py_bin) = py_bin else {
        eprintln!("[Tauri] Could not find the bundled or PATH Python executable.");
        return;
    };

    println!("[Tauri] Starting Python backend ({:?}) on {}...", py_bin, addr);
    let mut cmd = Command::new(&py_bin);
    cmd.args(["-m", "backend.main"])
        .current_dir(&root)
        .env("HF_HOME", root.join("models").join("huggingface"))
        .env("HF_HUB_CACHE", root.join("models").join("huggingface").join("hub"))
        .env("TORCH_HOME", root.join("models").join("torch"));

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW = 0x08000000 to keep it backgrounded
        cmd.creation_flags(0x08000000);
    }

    match cmd.spawn() {
        Ok(child) => {
            println!("[Tauri] Python backend spawned (PID: {}).", child.id());
            if let Ok(mut lock) = BACKEND_PROCESS.lock() {
                *lock = Some(child);
            }
        }
        Err(err) => {
            eprintln!("[Tauri] Could not auto-spawn Python backend: {}", err);
        }
    }
}

#[tauri::command]
fn quit_app(app: tauri::AppHandle) {
    app.exit(0);
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    ensure_backend_running();

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![quit_app])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|_app_handle, event| {
        if let RunEvent::ExitRequested { .. } = event {
            if let Ok(mut lock) = BACKEND_PROCESS.lock() {
                if let Some(mut child) = lock.take() {
                    println!("[Tauri] Terminating backend process...");
                    let _ = child.kill();
                }
            }
        }
    });
}
