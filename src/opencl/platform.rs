use ocl::{Context, Platform};

use crate::{Error, PlatformInstance};

const GPU_MIN_DEFAULT: usize = 1024; // 1 KiB

const ACC_MIN_DEFAULT: usize = 2_147_483_648; // 1 GiB

#[derive(Clone)]
struct DeviceList {
    devices: Vec<ocl::Device>,
    next: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl Default for DeviceList {
    fn default() -> Self {
        Self {
            devices: Vec::default(),
            next: std::sync::Arc::new(Default::default()),
        }
    }
}

impl DeviceList {
    fn is_empty(&self) -> bool {
        self.devices.is_empty()
    }

    fn next(&self) -> Option<ocl::Device> {
        if self.devices.is_empty() {
            None
        } else {
            let idx = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.devices.get(idx % self.devices.len()).copied()
        }
    }
}

impl From<Vec<ocl::Device>> for DeviceList {
    fn from(devices: Vec<ocl::Device>) -> Self {
        Self {
            devices,
            next: std::sync::Arc::new(Default::default()),
        }
    }
}

impl FromIterator<ocl::Device> for DeviceList {
    fn from_iter<T: IntoIterator<Item = ocl::Device>>(iter: T) -> Self {
        Self::from(iter.into_iter().collect::<Vec<ocl::Device>>())
    }
}

#[derive(Clone)]
/// An OpenCL platform
pub struct OpenCL {
    cl_platform: Platform,
    cl_context: Context,
    cl_cpus: DeviceList,
    cl_gpus: DeviceList,
    cl_accs: DeviceList,
}

impl OpenCL {
    fn default() -> Result<Self, Error> {
        let cl_platform = Platform::first()?;
        Self::try_from(cl_platform).map_err(Error::from)
    }

    /// Borrow the OpenCL [`Context`] of this platform.
    pub fn context(&self) -> &Context {
        &self.cl_context
    }

    fn has_gpu(&self) -> bool {
        !self.cl_gpus.is_empty()
    }

    fn next_cpu(&self) -> Option<ocl::Device> {
        self.cl_cpus.next()
    }

    fn next_gpu(&self) -> Option<ocl::Device> {
        self.cl_gpus.next()
    }

    fn next_acc(&self) -> Option<ocl::Device> {
        self.cl_accs.next()
    }

    fn select_device(&self, size_hint: usize) -> Option<ocl::Device> {
        if size_hint < GPU_MIN_DEFAULT {
            self.next_cpu()
        } else if size_hint < ACC_MIN_DEFAULT {
            self.next_gpu().or_else(|| self.next_cpu())
        } else {
            self.next_acc()
                .or_else(|| self.next_gpu())
                .or_else(|| self.next_cpu())
        }
    }
}

impl TryFrom<Platform> for OpenCL {
    type Error = ocl::Error;

    fn try_from(cl_platform: Platform) -> Result<Self, Self::Error> {
        let cl_cpus = ocl::Device::list(cl_platform, Some(ocl::DeviceType::CPU))?;
        let cl_gpus = ocl::Device::list(cl_platform, Some(ocl::DeviceType::GPU))?;
        let cl_accs = ocl::Device::list(cl_platform, Some(ocl::DeviceType::ACCELERATOR))?;

        let cl_context = ocl::builders::ContextBuilder::new()
            .platform(cl_platform)
            .devices(&cl_cpus)
            .devices(&cl_gpus)
            .devices(&cl_accs)
            .build()?;

        Ok(Self {
            cl_cpus: cl_cpus.into(),
            cl_gpus: cl_gpus.into(),
            cl_accs: cl_accs.into(),
            cl_context,
            cl_platform,
        })
    }
}

impl PlatformInstance for OpenCL {}
