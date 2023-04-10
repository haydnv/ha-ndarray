use std::fmt;

pub enum Error {
    NotFound(String),
    #[cfg(feature = "opencl")]
    OclError(ocl::error::Error),
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "not found: {}", id),
            #[cfg(feature = "opencl")]
            Self::OclError(cause) => cause.fmt(f),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "not found: {}", id),
            #[cfg(feature = "opencl")]
            Self::OclError(cause) => cause.fmt(f),
        }
    }
}

impl std::error::Error for Error {}

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
        let mut devices = Vec::with_capacity(3);

        devices.push(Device::Main);

        #[cfg(feature = "opencl")]
        {
            for platform in ocl::Platform::list() {
                let host_type = ocl::DeviceType::CPU;
                for device in ocl::Device::list(platform, Some(host_type))? {
                    if device.is_available()? {
                        devices.push(Device::CPUCL(device));
                    }
                }

                let gpu_type = ocl::DeviceType::GPU | ocl::DeviceType::ACCELERATOR;
                for device in ocl::Device::list(platform, Some(gpu_type))? {
                    if device.is_available()? {
                        devices.push(Device::GPU(device));
                    }
                }
            }
        }

        Ok(devices)
    }

    #[cfg(feature = "opencl")]
    pub fn gpu(i: usize) -> Result<ocl::Device, Error> {
        let mut count = 0;
        for platform in ocl::Platform::list() {
            let gpu_type = ocl::DeviceType::GPU | ocl::DeviceType::ACCELERATOR;
            for device in ocl::Device::list(platform, Some(gpu_type))? {
                if device.is_available()? {
                    if count == i {
                        return Ok(device);
                    } else {
                        count += 1;
                    }
                }
            }
        }

        Err(Error::NotFound(format!("OpenCL device {}", i)))
    }
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Main => f.write_str("CPU (main device)"),
            #[cfg(feature = "opencl")]
            Self::CPUCL(device) => write!(f, "OpenCL CPU: {:?}", device),
            #[cfg(feature = "opencl")]
            Self::GPU(device) => write!(f, "OpenCL GPU: {:?}", device),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "opencl")]
    #[test]
    fn test_buffer_and_kernel() -> Result<(), Error> {
        use ocl::{Buffer, Context, EventList, Program, Kernel};

        static SRC: &'static str = r#"
    __kernel void multiply(__global float* left, __global float* right, __global float* output) {
        int id = get_global_id(0);
        output[id] = left[id] * right[id];
    }
"#;

        for device in Device::all().expect("device list") {
            println!("found device: {:?}", device);
        }

        let context = Context::builder().build().expect("context");
        let gpu = Device::gpu(0).expect("GPU");
        let queue = ocl::Queue::new(&context, gpu, None).expect("queue");

        let dims = 8;

        let left = Buffer::<f32>::builder()
            .queue(queue.clone())
            .len(dims)
            .build()?;

        let right = Buffer::<f32>::builder()
            .queue(queue.clone())
            .len(dims)
            .build()?;

        let output = Buffer::<f32>::builder()
            .queue(queue.clone())
            .len(dims)
            .build()?;

        let program = Program::builder()
            .src(SRC)
            .devices(gpu)
            .build(&context)?;

        let kernel = Kernel::builder()
            .name("multiply")
            .program(&program)
            .queue(queue.clone())
            .global_work_size(dims)
            .arg(&left)
            .arg(&right)
            .arg(&output)
            .build()?;

        let mut event_list = EventList::new();
        unsafe {
            kernel.cmd().enew(&mut event_list).enq()?;
        }
        event_list.wait_for().map_err(Error::from)
    }
}
