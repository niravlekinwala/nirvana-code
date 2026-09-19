use std::process::Command;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SiliconProfile {
    pub chip_name: String,
    pub p_cores: u32,
    pub e_cores: u32,
    pub gpu_cores: u32,
    pub memory_gb: u32,
    pub memory_bandwidth_gbps: u32,
    pub metal_version: String,
}

impl Default for SiliconProfile {
    fn default() -> Self {
        Self::detect()
    }
}

impl SiliconProfile {
    /// Probe once per process; callers on the generation path hit this per request.
    pub fn detect() -> Self {
        static PROFILE: std::sync::OnceLock<SiliconProfile> = std::sync::OnceLock::new();
        PROFILE.get_or_init(Self::probe).clone()
    }

    fn probe() -> Self {
        let chip_name = Command::new("sysctl")
            .arg("-n")
            .arg("machdep.cpu.brand_string")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "Apple Silicon (M2 Pro)".to_string());

        let total_mem_bytes = Command::new("sysctl")
            .arg("-n")
            .arg("hw.memsize")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(17_179_869_184); // Default 16 GB

        let memory_gb = (total_mem_bytes / (1024 * 1024 * 1024)) as u32;

        // Apple M2 Pro specs
        let (p_cores, e_cores, gpu_cores, memory_bandwidth_gbps) = if chip_name.contains("M2 Pro") {
            (8, 4, 16, 200)
        } else if chip_name.contains("M2 Max") {
            (8, 4, 30, 400)
        } else if chip_name.contains("M3 Pro") {
            (6, 6, 18, 150)
        } else if chip_name.contains("M3 Max") {
            (12, 4, 30, 300)
        } else if chip_name.contains("M4") {
            (8, 4, 16, 273)
        } else if chip_name.contains("M1 Pro") {
            (8, 2, 16, 200)
        } else if chip_name.contains("M1 Max") {
            (8, 2, 32, 400)
        } else {
            (8, 4, 16, 200)
        };

        Self {
            chip_name,
            p_cores,
            e_cores,
            gpu_cores,
            memory_gb,
            memory_bandwidth_gbps,
            metal_version: "Metal 3 (Unified Memory)".to_string(),
        }
    }
}
