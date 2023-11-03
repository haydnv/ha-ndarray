use std::borrow::Borrow;
use std::sync::Arc;

use ocl::core::{DeviceInfo, DeviceInfoResult};
use ocl::{Buffer, Context, Device, DeviceType, Platform, Queue};

use crate::{CType, Error, PlatformInstance};

use super::CL_PLATFORM;

pub const GPU_MIN_SIZE: usize = 1024; // 1 KiB

pub const ACC_MIN_SIZE: usize = 2_147_483_648; // 1 GiB

#[derive(Clone)]
struct DeviceList {
    devices: Vec<Device>,
    next: Arc<std::sync::atomic::AtomicUsize>,
}

impl Default for DeviceList {
    fn default() -> Self {
        Self {
            devices: Vec::default(),
            next: Arc::new(Default::default()),
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

pub struct CLPlatform {
    cl_platform: Platform,
    cl_context: Context,
    cl_cpus: DeviceList,
    cl_gpus: DeviceList,
    cl_accs: DeviceList,
}

impl CLPlatform {
    pub(super) fn default() -> Result<Self, Error> {
        let cl_platform = Platform::first()?;
        Self::try_from(cl_platform).map_err(Error::from)
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
        if size_hint < GPU_MIN_SIZE {
            DeviceType::CPU
        } else if size_hint < ACC_MIN_SIZE {
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

impl TryFrom<Platform> for CLPlatform {
    type Error = ocl::Error;

    fn try_from(cl_platform: Platform) -> Result<Self, Self::Error> {
        let devices = Device::list(cl_platform, None)?;
        let cl_context = ocl::builders::ContextBuilder::new()
            .platform(cl_platform)
            .devices(&devices)
            .build()?;

        let cl_cpus = Device::list(cl_platform, Some(DeviceType::CPU))?;
        let cl_gpus = Device::list(cl_platform, Some(DeviceType::GPU))?;
        let cl_accs = Device::list(cl_platform, Some(DeviceType::ACCELERATOR))?;

        Ok(Self {
            cl_cpus: cl_cpus.into(),
            cl_gpus: cl_gpus.into(),
            cl_accs: cl_accs.into(),
            cl_context,
            cl_platform,
        })
    }
}

/// The OpenCL platform
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct OpenCL;

impl PlatformInstance for OpenCL {
    fn select(_size_hint: usize) -> Self {
        Self
    }
}

pub trait CLBuf<T: CType>: Borrow<Buffer<T>> {}

impl OpenCL {
    /// Borrow the OpenCL [`Context`] of this platform.
    pub fn context(&self) -> &Context {
        &CL_PLATFORM.cl_context
    }

    /// Copy the given `data` into a new [`Buffer`].
    pub fn copy_into_buffer<T: CType>(data: &[T]) -> Result<Buffer<T>, ocl::Error> {
        ocl::builders::BufferBuilder::new()
            .len(data.len())
            .context(&CL_PLATFORM.cl_context)
            .copy_host_slice(data)
            .build()
    }

    /// Create a new [`Buffer`].
    pub fn create_buffer<T: CType>(size: usize) -> Result<Buffer<T>, ocl::Error> {
        ocl::builders::BufferBuilder::new()
            .len(size)
            .context(&CL_PLATFORM.cl_context)
            .build()
    }

    pub(crate) fn queue(
        &self,
        size_hint: usize,
        left: Option<&Queue>,
        right: Option<&Queue>,
    ) -> Result<Queue, ocl::Error> {
        let device_type = CL_PLATFORM.select_device_type(size_hint);

        // TODO: is this slow?
        if let Some(queue) = left {
            if let DeviceInfoResult::Type(dt) = queue.device().info(DeviceInfo::Type)? {
                if dt == device_type {
                    return Ok(queue.clone());
                }
            }
        }

        // TODO: is this slow?
        if let Some(queue) = right {
            if let DeviceInfoResult::Type(dt) = queue.device().info(DeviceInfo::Type)? {
                if dt == device_type {
                    return Ok(queue.clone());
                }
            }
        }

        let device = CL_PLATFORM
            .select_device(device_type)
            .expect("OpenCL device");

        Queue::new(self.context(), device, None)
    }
}
