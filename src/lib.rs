pub enum Error {
    #[cfg(feature = "opencl")]
    OclError(ocl::error::Error),
}

#[cfg(feature = "opencl")]
impl From<ocl::error::Error> for Error {
    fn from(cause: ocl::Error) -> Self {
        Self::OclError(cause)
    }
}

pub enum Device {
    Main,
    #[cfg(feature = "opencl")]
    CPUCL(ocl::Device),
    #[cfg(feature = "opencl")]
    GPU(ocl::Device),
}

impl Device {
    pub fn all() -> Result<Vec<Device>, Error> {
        let mut devices = Vec::with_capacity(2);

        devices.push(Device::Main);

        #[cfg(feature = "opencl")]
        {
            for platform in ocl::Platform::list() {
                let host_type = ocl::DeviceType::CPU;
                for device in ocl::Device::list(platform, Some(host_type))? {
                    devices.push(Device::CPUCL(device));
                }

                let gpu_type = ocl::DeviceType::GPU | ocl::DeviceType::ACCELERATOR;
                for device in ocl::Device::list(platform, Some(gpu_type))? {
                    devices.push(Device::GPU(device));
                }
            }
        }

        Ok(devices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "opencl")]
    #[test]
    fn test_create_buffer() {
        use ocl::builders::{BufferBuilder, ContextBuilder};

        let context = ContextBuilder::new().build().expect("context");
        let _buffer = BufferBuilder::<f32>::new().context(&context).len(8).build();
    }
}
