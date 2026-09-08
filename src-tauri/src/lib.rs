use std::env;
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;
use tauri::RunEvent;

const BACKEND_PORT: u16 = 48653;
const VITE_PORTS: [u16; 2] = [41765, 41766];

struct ManagedProcess {
    pid: u32,
    child: Option<Child>,
}

static BACKEND_PROCESS: Mutex<Option<ManagedProcess>> = Mutex::new(None);
static CLEANUP_DONE: AtomicBool = AtomicBool::new(false);

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

fn silent_command(program: &str) -> Command {
    let mut cmd = Command::new(program);
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    cmd
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

fn listening_pids_for_ports(ports: &[u16]) -> Vec<u32> {
    let output = match silent_command("netstat").args(["-ano", "-p", "tcp"]).output() {
        Ok(output) => output,
        Err(_) => return Vec::new(),
    };

    let suffixes: Vec<String> = ports.iter().map(|p| format!(":{}", p)).collect();
    let mut pids = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        // Windows netstat format: TCP local-address remote-address state pid.
        if fields.len() >= 5
            && fields[0].eq_ignore_ascii_case("TCP")
            && suffixes.iter().any(|suffix| fields[1].ends_with(suffix))
            && fields[3].eq_ignore_ascii_case("LISTENING")
        {
            if let Ok(pid) = fields[4].parse::<u32>() {
                pids.push(pid);
            }
        }
    }
    pids.sort_unstable();
    pids.dedup();
    pids
}

fn listening_pids(port: u16) -> Vec<u32> {
    listening_pids_for_ports(&[port])
}

fn kill_process_tree(pid: u32) {
    if pid == 0 {
        return;
    }

    #[cfg(target_os = "windows")]
    {
        // /T is important: Python/Node may have spawned children that otherwise
        // survive the parent process. /F deliberately makes shutdown reliable.
        let _ = silent_command("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .output();
    }

    #[cfg(not(target_os = "windows"))]
    {
        let _ = Command::new("kill")
            .args(["-KILL", &pid.to_string()])
            .output();
    }
}

fn terminate_process(process: ManagedProcess) {
    let pid = process.pid;
    if let Some(mut child) = process.child {
        let _ = child.kill();
        let _ = child.wait();
    }
    kill_process_tree(pid);
}

fn cleanup_processes() {
    if CLEANUP_DONE.swap(true, Ordering::SeqCst) {
        return;
    }

    let managed_backend = BACKEND_PROCESS.lock().ok().and_then(|mut lock| lock.take());
    if let Some(process) = managed_backend {
        terminate_process(process);
    }

    // Also reap processes left by an older launcher/crashed frontend. These are
    // application-reserved ports, so this catches stale Python and Vite servers
    // even when this Tauri instance did not spawn them itself.
    let all_ports: Vec<u16> = std::iter::once(BACKEND_PORT).chain(VITE_PORTS).collect();
    let pids = listening_pids_for_ports(&all_ports);
    for pid in pids {
        kill_process_tree(pid);
    }
}

fn ensure_backend_running() {
    let addr = format!("127.0.0.1:{}", BACKEND_PORT);
    if TcpStream::connect_timeout(&addr.parse().unwrap(), Duration::from_millis(250)).is_ok() {
        let pid = listening_pids(BACKEND_PORT).into_iter().next();
        println!(
            "[Tauri] Backend is already running on {} (PID: {:?}).",
            addr, pid
        );
        if let Ok(mut lock) = BACKEND_PROCESS.lock() {
            *lock = pid.map(|pid| ManagedProcess { pid, child: None });
        }
        return;
    }

    let root = packaged_root();
    let py_bin = local_python(&root).or_else(|| {
        for name in ["pythonw", "python"] {
            if silent_command(name).arg("--version").output().is_ok() {
                return Some(PathBuf::from(name));
            }
        }
        None
    });

    let Some(py_bin) = py_bin else {
        eprintln!("[Tauri] Could not find the bundled or PATH Python executable.");
        return;
    };

    println!(
        "[Tauri] Starting Python backend ({:?}) on {}...",
        py_bin, addr
    );
    let mut cmd = silent_command(py_bin.to_str().unwrap_or("python"));
    cmd.args(["-m", "backend.main"])
        .current_dir(&root);

    let is_portable = root.join("python").is_dir();
    if is_portable && root.join("models").is_dir() {
        cmd.env("HF_HOME", root.join("models").join("huggingface"))
            .env(
                "HF_HUB_CACHE",
                root.join("models").join("huggingface").join("hub"),
            )
            .env("TORCH_HOME", root.join("models").join("torch"));
    }

    match cmd.spawn() {
        Ok(child) => {
            let pid = child.id();
            println!("[Tauri] Python backend spawned (PID: {}).", pid);
            if let Ok(mut lock) = BACKEND_PROCESS.lock() {
                *lock = Some(ManagedProcess {
                    pid,
                    child: Some(child),
                });
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
        if matches!(event, RunEvent::ExitRequested { .. } | RunEvent::Exit) {
            println!("[Tauri] Hard-stopping Sayso child processes...");
            cleanup_processes();
        }
    });
}
