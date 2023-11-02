use ocl::core::{DeviceInfo, DeviceInfoResult};
use ocl::{Context, Device, DeviceType, Platform, Queue};

use crate::{Error, PlatformInstance};

const GPU_MIN_DEFAULT: usize = 1024; // 1 KiB

const ACC_MIN_DEFAULT: usize = 2_147_483_648; // 1 GiB

#[derive(Clone)]
struct DeviceList {
    devices: Vec<Device>,
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

    fn next(&self) -> Option<Device> {
        if self.devices.is_empty() {
            None
        } else {
            let idx = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.devices.get(idx % self.devices.len()).copied()
        }
    }
}

impl From<Vec<Device>> for DeviceList {
    fn from(devices: Vec<Device>) -> Self {
        Self {
            devices,
            next: std::sync::Arc::new(Default::default()),
        }
    }
}

impl FromIterator<Device> for DeviceList {
    fn from_iter<T: IntoIterator<Item = Device>>(iter: T) -> Self {
        Self::from(iter.into_iter().collect::<Vec<Device>>())
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

    pub(crate) fn queue(
        &self,
        size_hint: usize,
        left: Option<&Queue>,
        right: Option<&Queue>,
    ) -> Result<Queue, ocl::Error> {
        let device_type = self.select_device_type(size_hint);

        if let Some(queue) = left {
            if let DeviceInfoResult::Type(dt) = queue.device().info(DeviceInfo::Type)? {
                if dt == device_type {
                    return Ok(queue.clone());
                }
            }
        }

        if let Some(queue) = right {
            if let DeviceInfoResult::Type(dt) = queue.device().info(DeviceInfo::Type)? {
                if dt == device_type {
                    return Ok(queue.clone());
                }
            }
        }

        let device = self.select_device(device_type).expect("OpenCL device");
        Queue::new(self.context(), device, None)
    }

    fn next_cpu(&self) -> Option<Device> {
        self.cl_cpus.next()
    }

    fn next_gpu(&self) -> Option<Device> {
        self.cl_gpus.next()
    }

    fn next_acc(&self) -> Option<Device> {
        self.cl_accs.next()
    }

    fn select_device_type(&self, size_hint: usize) -> DeviceType {
        if size_hint < GPU_MIN_DEFAULT {
            DeviceType::CPU
        } else if size_hint < ACC_MIN_DEFAULT {
            DeviceType::GPU
        } else {
            DeviceType::ACCELERATOR
        }
    }

    fn select_device(&self, device_type: DeviceType) -> Option<Device> {
        match device_type {
            DeviceType::CPU => self
                .next_cpu()
                .or_else(|| self.next_gpu())
                .or_else(|| self.next_acc()),

            DeviceType::GPU => self
                .next_gpu()
                .or_else(|| self.next_acc())
                .or_else(|| self.next_cpu()),

            DeviceType::ACCELERATOR => self
                .next_acc()
                .or_else(|| self.next_gpu())
                .or_else(|| self.next_cpu()),

            other => panic!("unsupported OpenCL device type: {other:?}"),
        }
    }
}

impl TryFrom<Platform> for OpenCL {
    type Error = ocl::Error;

    fn try_from(cl_platform: Platform) -> Result<Self, Self::Error> {
        let cl_cpus = Device::list(cl_platform, Some(ocl::DeviceType::CPU))?;
        let cl_gpus = Device::list(cl_platform, Some(ocl::DeviceType::GPU))?;
        let cl_accs = Device::list(cl_platform, Some(ocl::DeviceType::ACCELERATOR))?;

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
