//! Apple Silicon hardware probe.
//!
//! Everything here is read directly from the kernel (`sysctlbyname`) and the
//! I/O registry, not by spawning processes or matching marketing names. The
//! probe runs once per process; every caller on the generation path goes
//! through the cached copy.

use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::OnceLock;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SiliconProfile {
    pub chip_name: String,
    pub p_cores: u32,
    pub e_cores: u32,
    pub gpu_cores: u32,
    pub memory_gb: u32,
    pub memory_bytes: u64,
    /// How much unified memory the GPU may wire down. Reads the
    /// `iogpu.wired_limit_mb` override when set, otherwise macOS's default
    /// split (two thirds up to 36 GB, three quarters above).
    pub gpu_wired_limit_bytes: u64,
    pub gpu_wired_limit_is_override: bool,
    /// Informational; from a lookup table, not measured.
    pub memory_bandwidth_gbps: u32,
    pub metal_version: String,
}

impl Default for SiliconProfile {
    fn default() -> Self {
        Self::detect()
    }
}

impl SiliconProfile {
    /// Probe once per process.
    pub fn detect() -> Self {
        static PROFILE: OnceLock<SiliconProfile> = OnceLock::new();
        PROFILE.get_or_init(Self::probe).clone()
    }

    fn probe() -> Self {
        let chip_name = sysctl_string("machdep.cpu.brand_string")
            .unwrap_or_else(|| "Apple Silicon".to_string());

        // perflevel0 is the performance cluster, perflevel1 the efficiency
        // cluster. Intel Macs and VMs report neither; fall back to the total.
        let p_cores = sysctl_u64("hw.perflevel0.physicalcpu")
            .or_else(|| sysctl_u64("hw.physicalcpu"))
            .unwrap_or(4) as u32;
        let e_cores = sysctl_u64("hw.perflevel1.physicalcpu").unwrap_or(0) as u32;

        let memory_bytes = sysctl_u64("hw.memsize").unwrap_or(16 << 30);
        let memory_gb = (memory_bytes >> 30) as u32;

        let wired_override_mb = sysctl_u64("iogpu.wired_limit_mb").unwrap_or(0);
        let (gpu_wired_limit_bytes, gpu_wired_limit_is_override) = if wired_override_mb > 0 {
            (wired_override_mb << 20, true)
        } else if memory_bytes <= 36 << 30 {
            (memory_bytes / 3 * 2, false)
        } else {
            (memory_bytes / 4 * 3, false)
        };

        let gpu_cores = gpu_core_count().unwrap_or(0);

        Self {
            memory_bandwidth_gbps: bandwidth_from_name(&chip_name),
            chip_name,
            p_cores,
            e_cores,
            gpu_cores,
            memory_gb,
            memory_bytes,
            gpu_wired_limit_bytes,
            gpu_wired_limit_is_override,
            metal_version: "Metal 3 (Unified Memory)".to_string(),
        }
    }

    /// Suggested `sysctl iogpu.wired_limit_mb` so a model of `model_bytes`
    /// plus working set fits in GPU-wired memory. `None` when it already fits
    /// or when raising the limit would leave the OS with under 3 GB.
    pub fn suggested_wired_limit_mb(&self, model_bytes: u64) -> Option<u64> {
        let needed = model_bytes + model_bytes / 5 + (1 << 30); // weights + ~20% KV/compute + 1 GB
        if needed <= self.gpu_wired_limit_bytes {
            return None;
        }
        let ceiling = self.memory_bytes.saturating_sub(3 << 30);
        if needed > ceiling {
            return None;
        }
        Some(needed >> 20)
    }
}

/// Nominal LPDDR bandwidth by family/tier. Only used for display.
fn bandwidth_from_name(name: &str) -> u32 {
    let n = name.to_ascii_lowercase();
    let tier = if n.contains("ultra") {
        3
    } else if n.contains("max") {
        2
    } else if n.contains("pro") {
        1
    } else {
        0
    };
    let family = ["m5", "m4", "m3", "m2", "m1"]
        .iter()
        .find(|g| n.contains(*g))
        .copied()
        .unwrap_or("m2");
    match (family, tier) {
        ("m1", 0) => 68,
        ("m1", 1) => 200,
        ("m1", 2) => 400,
        ("m1", 3) => 800,
        ("m2", 0) => 100,
        ("m2", 1) => 200,
        ("m2", 2) => 400,
        ("m2", 3) => 800,
        ("m3", 0) => 100,
        ("m3", 1) => 150,
        ("m3", 2) => 400,
        ("m3", 3) => 800,
        ("m4", 0) => 120,
        ("m4", 1) => 273,
        ("m4", 2) => 546,
        ("m4", 3) => 819,
        ("m5", 0) => 153,
        ("m5", 1) => 300,
        ("m5", 2) => 600,
        _ => 800,
    }
}

fn sysctl_u64(name: &str) -> Option<u64> {
    let cname = CString::new(name).ok()?;
    let mut value: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let rc = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            (&mut value as *mut u64).cast::<c_void>(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return None;
    }
    // Some keys are 32-bit; the kernel writes `len` bytes into the buffer.
    Some(if len == 4 { value & 0xffff_ffff } else { value })
}

fn sysctl_string(name: &str) -> Option<String> {
    let cname = CString::new(name).ok()?;
    let mut len = 0usize;
    let rc = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            std::ptr::null_mut(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 || len == 0 {
        return None;
    }
    let mut buf = vec![0u8; len];
    let rc = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            buf.as_mut_ptr().cast::<c_void>(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return None;
    }
    let s = CStr::from_bytes_until_nul(&buf).ok()?;
    Some(s.to_string_lossy().trim().to_string())
}

/// GPU core count from the `AGXAccelerator` entry in the I/O registry.
#[cfg(target_os = "macos")]
fn gpu_core_count() -> Option<u32> {
    type CFTypeRef = *const c_void;
    type CFStringRef = *const c_void;
    type CFDictionaryRef = *const c_void;
    type IoObject = u32;

    #[link(name = "IOKit", kind = "framework")]
    unsafe extern "C" {
        fn IOServiceMatching(name: *const c_char) -> CFDictionaryRef;
        fn IOServiceGetMatchingService(main_port: u32, matching: CFDictionaryRef) -> IoObject;
        fn IORegistryEntryCreateCFProperty(
            entry: IoObject,
            key: CFStringRef,
            allocator: *const c_void,
            options: u32,
        ) -> CFTypeRef;
        fn IOObjectRelease(object: IoObject) -> i32;
    }
    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFStringCreateWithCString(
            alloc: *const c_void,
            s: *const c_char,
            encoding: u32,
        ) -> CFStringRef;
        fn CFNumberGetValue(number: CFTypeRef, the_type: i64, value_ptr: *mut c_void) -> bool;
        fn CFRelease(cf: CFTypeRef);
    }
    const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
    const K_CF_NUMBER_SINT32_TYPE: i64 = 3;

    unsafe {
        let name = CString::new("AGXAccelerator").ok()?;
        let matching = IOServiceMatching(name.as_ptr());
        if matching.is_null() {
            return None;
        }
        // Consumes `matching`.
        let service = IOServiceGetMatchingService(0, matching);
        if service == 0 {
            return None;
        }
        let key_c = CString::new("gpu-core-count").ok()?;
        let key =
            CFStringCreateWithCString(std::ptr::null(), key_c.as_ptr(), K_CF_STRING_ENCODING_UTF8);
        let prop = IORegistryEntryCreateCFProperty(service, key, std::ptr::null(), 0);
        let mut cores: i32 = 0;
        let ok = !prop.is_null()
            && CFNumberGetValue(
                prop,
                K_CF_NUMBER_SINT32_TYPE,
                (&mut cores as *mut i32).cast::<c_void>(),
            );
        if !prop.is_null() {
            CFRelease(prop);
        }
        CFRelease(key);
        IOObjectRelease(service);
        (ok && cores > 0).then_some(cores as u32)
    }
}

#[cfg(not(target_os = "macos"))]
fn gpu_core_count() -> Option<u32> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_reads_real_values() {
        let p = SiliconProfile::detect();
        eprintln!("{p:#?}");
        assert!(p.p_cores >= 1);
        assert!(p.memory_bytes >= 1 << 30);
        assert!(p.gpu_wired_limit_bytes < p.memory_bytes);
        assert!(!p.chip_name.is_empty());
    }

    #[test]
    fn wired_limit_suggestion() {
        let p = SiliconProfile {
            chip_name: "Apple M2 Pro".into(),
            p_cores: 8,
            e_cores: 4,
            gpu_cores: 16,
            memory_gb: 16,
            memory_bytes: 16 << 30,
            gpu_wired_limit_bytes: (16u64 << 30) / 3 * 2,
            gpu_wired_limit_is_override: false,
            memory_bandwidth_gbps: 200,
            metal_version: String::new(),
        };
        assert_eq!(p.suggested_wired_limit_mb(2 << 30), None); // 2 GB model fits
        let mb = p.suggested_wired_limit_mb(9 << 30).unwrap(); // 9 GB model needs ~11.8 GB
        assert!(mb > 10 * 1024 && mb < 13 * 1024, "{mb}");
        assert_eq!(p.suggested_wired_limit_mb(14 << 30), None); // would starve the OS
    }

    #[test]
    fn bandwidth_table_covers_families() {
        assert_eq!(bandwidth_from_name("Apple M2 Pro"), 200);
        assert_eq!(bandwidth_from_name("Apple M4 Max"), 546);
        assert_eq!(bandwidth_from_name("Apple M1"), 68);
    }
}
