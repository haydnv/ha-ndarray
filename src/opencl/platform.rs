use std::sync::Arc;

use ocl::core::{DeviceInfo, DeviceInfoResult};
use ocl::{Buffer, Context, Device, DeviceType, Kernel, Platform, Queue};
use rayon::prelude::*;

use crate::access::{Access, AccessOp};
use crate::buffer::BufferConverter;
use crate::ops::{
    Cond, Construct, ElementwiseBoolean, ElementwiseCompare, ElementwiseDual,
    ElementwiseScalarCompare, ElementwiseUnary, LinAlgDual, Random, ReduceAll, ReduceAxis,
    Transform,
};
use crate::platform::{Convert, PlatformInstance};
use crate::{Axes, CType, Constant, Error, Float, Range, Shape};

use super::ops::*;
use super::{programs, CLConverter};
use super::{CL_PLATFORM, WG_SIZE};

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
            next: Arc::new(Default::default()),
        }
    }
}

impl FromIterator<Device> for DeviceList {
    fn from_iter<T: IntoIterator<Item = Device>>(iter: T) -> Self {
        Self::from(iter.into_iter().collect::<Vec<Device>>())
    }
}

pub struct CLPlatform {
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
        let cl_context = Context::builder()
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
        })
    }
}

/// The OpenCL platform
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct OpenCL;

impl OpenCL {
    /// Borrow the OpenCL [`Context`] of this platform.
    pub fn context<'a>() -> &'a Context {
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

    pub(crate) fn queue(
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

        Queue::new(&CL_PLATFORM.cl_context, device, None)
    }
}

impl PlatformInstance for OpenCL {
    fn select(_size_hint: usize) -> Self {
        Self
    }
}

impl<T: CType> Constant<T> for OpenCL {
    type Buffer = Buffer<T>;

    fn constant(&self, value: T, size: usize) -> Result<Self::Buffer, Error> {
        let queue = Self::queue(size, None, None)?;

        ocl::builders::BufferBuilder::new()
            .len(size)
            .fill_val(value)
            .queue(queue)
            .build()
            .map_err(Error::from)
    }
}

impl<'a, T: CType> Convert<'a, T> for OpenCL {
    type Buffer = CLConverter<'a, T>;

    fn convert(&self, buffer: BufferConverter<'a, T>) -> Result<Self::Buffer, Error> {
        buffer.to_cl().map_err(Error::from)
    }
}

impl<T: CType> Construct<T> for OpenCL {
    type Range = Linear<T>;

    fn range(self, start: T, stop: T, size: usize) -> Result<AccessOp<Self::Range, Self>, Error> {
        if start <= stop {
            let step = (stop - start).to_float().to_f64() / size as f64;
            Linear::new(start, step, size).map(AccessOp::from)
        } else {
            Err(Error::Bounds(format!("invalid range: [{start}, {stop})")))
        }
    }
}

impl<A, L, R, T> Cond<A, L, R, T> for OpenCL
where
    A: Access<u8>,
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Op = GatherCond<A, L, R, T>;

    fn cond(self, cond: A, then: L, or_else: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        GatherCond::new(cond, then, or_else).map(AccessOp::from)
    }
}

impl<T, L, R> ElementwiseBoolean<L, R, T> for OpenCL
where
    T: CType,
    L: Access<T>,
    R: Access<T>,
{
    type Op = Dual<L, R, T, u8>;

    fn and(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Dual::and(left, right).map(AccessOp::from)
    }

    fn or(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Dual::or(left, right).map(AccessOp::from)
    }

    fn xor(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Dual::xor(left, right).map(AccessOp::from)
    }
}

impl<T, L, R> ElementwiseCompare<L, R, T> for OpenCL
where
    T: CType,
    L: Access<T>,
    R: Access<T>,
{
    type Op = Dual<L, R, T, u8>;

    fn eq(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Dual::eq(left, right).map(AccessOp::from)
    }

    fn ge(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Dual::ge(left, right).map(AccessOp::from)
    }

    fn gt(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Dual::gt(left, right).map(AccessOp::from)
    }

    fn le(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Dual::le(left, right).map(AccessOp::from)
    }

    fn lt(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Dual::lt(left, right).map(AccessOp::from)
    }

    fn ne(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Dual::ne(left, right).map(AccessOp::from)
    }
}

impl<A: Access<T>, T: CType> ElementwiseScalarCompare<A, T> for OpenCL {
    type Op = Scalar<A, T, u8>;

    fn eq_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Scalar::eq(left, right).map(AccessOp::from)
    }

    fn ge_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Scalar::ge(left, right).map(AccessOp::from)
    }

    fn gt_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Scalar::gt(left, right).map(AccessOp::from)
    }

    fn le_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Scalar::le(left, right).map(AccessOp::from)
    }

    fn lt_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Scalar::lt(left, right).map(AccessOp::from)
    }

    fn ne_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        Scalar::ne(left, right).map(AccessOp::from)
    }
}

impl<T, L, R> ElementwiseDual<L, R, T> for OpenCL
where
    T: CType,
    L: Access<T>,
    R: Access<T>,
{
    type Op = Dual<L, R, T, T>;

    fn add(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Dual::add(left, right).map(AccessOp::from)
    }

    fn sub(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Dual::sub(left, right).map(AccessOp::from)
    }
}

impl<A: Access<T>, T: CType> ElementwiseUnary<A, T> for OpenCL {
    type Op = Unary<A, T, T>;

    fn ln(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Unary::ln(access).map(AccessOp::from)
    }
}

impl<L, R, T> LinAlgDual<L, R, T> for OpenCL
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Op = MatMul<L, R, T>;

    fn matmul(
        self,
        left: L,
        right: R,
        dims: [usize; 4],
    ) -> Result<AccessOp<Self::Op, Self>, Error> {
        MatMul::new(left, right, dims).map(AccessOp::from)
    }
}

impl Random for OpenCL {
    type Normal = RandomNormal;
    type Uniform = RandomUniform;

    fn random_normal(self, size: usize) -> Result<AccessOp<Self::Normal, Self>, Error> {
        RandomNormal::new(size).map(AccessOp::from)
    }

    fn random_uniform(self, size: usize) -> Result<AccessOp<Self::Uniform, Self>, Error> {
        RandomUniform::new(size).map(AccessOp::from)
    }
}

impl<A: Access<T>, T: CType> ReduceAll<A, T> for OpenCL {
    fn all(self, access: A) -> Result<bool, Error> {
        let buffer = access.read()?.to_cl()?;

        let program = programs::reduce::all(T::TYPE)?;

        let flag = self.constant(1, 1)?;

        let queue = Self::queue(buffer.len(), buffer.default_queue(), None)?;

        let kernel = Kernel::builder()
            .name("all")
            .program(&program)
            .queue(queue.clone())
            .global_work_size(buffer.len())
            .arg(&flag)
            .arg(&*buffer)
            .build()?;

        unsafe { kernel.enq()? }

        let mut result = [1];
        flag.read(result.as_mut_slice()).enq()?;
        Ok(result == [1])
    }

    fn any(self, access: A) -> Result<bool, Error> {
        let buffer = access.read()?.to_cl()?;

        let program = programs::reduce::any(T::TYPE)?;

        let flag = self.constant(0, 1)?;

        let queue = Self::queue(buffer.len(), buffer.default_queue(), None)?;

        let kernel = Kernel::builder()
            .name("any")
            .program(&program)
            .queue(queue.clone())
            .global_work_size(buffer.len())
            .arg(&flag)
            .arg(&*buffer)
            .build()?;

        unsafe { kernel.enq()? }

        let mut result = [0];
        flag.read(result.as_mut_slice()).enq()?;
        Ok(result == [1])
    }

    fn max(self, access: A) -> Result<T, Error> {
        let input = access.read()?.to_cl()?;
        let result = reduce_all(&*input, "max", T::MIN)?;
        Ok(result.into_par_iter().reduce(|| T::MIN, T::max))
    }

    fn min(self, access: A) -> Result<T, Error> {
        let input = access.read()?.to_cl()?;
        let result = reduce_all(&*input, "min", T::MAX)?;
        Ok(result.into_par_iter().reduce(|| T::MAX, T::min))
    }

    fn product(self, access: A) -> Result<T, Error> {
        let input = access.read()?.to_cl()?;
        let result = reduce_all(&*input, "mul", T::ONE)?;
        Ok(result.into_par_iter().product())
    }

    fn sum(self, access: A) -> Result<T, Error> {
        let input = access.read()?.to_cl()?;
        let result = reduce_all(&*input, "add", T::ZERO)?;
        Ok(result.into_par_iter().sum())
    }
}

impl<A: Access<T>, T: CType> ReduceAxis<A, T> for OpenCL {
    type Op = Reduce<A, T>;

    fn max(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        Reduce::max(access, stride).map(AccessOp::from)
    }

    fn min(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        Reduce::min(access, stride).map(AccessOp::from)
    }

    fn product(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        Reduce::product(access, stride).map(AccessOp::from)
    }

    fn sum(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        Reduce::sum(access, stride).map(AccessOp::from)
    }
}

impl<A: Access<T>, T: CType> Transform<A, T> for OpenCL {
    type Broadcast = View<A, T>;
    type Slice = Slice<A, T>;
    type Transpose = View<A, T>;

    fn broadcast(
        self,
        access: A,
        shape: Shape,
        broadcast: Shape,
    ) -> Result<AccessOp<Self::Broadcast, Self>, Error> {
        View::broadcast(access, shape, broadcast).map(AccessOp::from)
    }

    fn slice(
        self,
        access: A,
        shape: &[usize],
        range: Range,
    ) -> Result<AccessOp<Self::Slice, Self>, Error> {
        Slice::new(access, shape, range).map(AccessOp::from)
    }

    fn transpose(
        self,
        access: A,
        shape: Shape,
        permutation: Axes,
    ) -> Result<AccessOp<Self::Transpose, Self>, Error> {
        View::transpose(access, shape, permutation).map(AccessOp::from)
    }
}

fn reduce_all<T: CType>(input: &Buffer<T>, reduce: &'static str, id: T) -> Result<Vec<T>, Error> {
    const MIN_SIZE: usize = 8192;

    let min_size = MIN_SIZE * num_cpus::get();

    let queue = OpenCL::queue(input.len(), input.default_queue(), None)?;

    if input.len() < min_size {
        let mut result = vec![id; input.len()];
        input.read(&mut result).queue(&queue).enq()?;
        return Ok(result);
    }

    let program = programs::reduce::reduce(T::TYPE, reduce)?;

    let mut buffer = {
        let output = Buffer::builder()
            .queue(queue.clone())
            .len(input.len().div_ceil(WG_SIZE))
            .fill_val(id)
            .build()?;

        let kernel = Kernel::builder()
            .name("reduce")
            .program(&program)
            .queue(queue.clone())
            .local_work_size(WG_SIZE)
            .global_work_size(WG_SIZE * output.len())
            .arg(input.len() as u64)
            .arg(&*input)
            .arg(&output)
            .arg_local::<T>(WG_SIZE)
            .build()?;

        unsafe { kernel.enq()? };

        output
    };

    while buffer.len() >= min_size {
        let input = buffer;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(input.len().div_ceil(WG_SIZE))
            .fill_val(id)
            .build()?;

        let kernel = Kernel::builder()
            .name("reduce")
            .program(&program)
            .queue(queue.clone())
            .local_work_size(WG_SIZE)
            .global_work_size(WG_SIZE * output.len())
            .arg(input.len() as u64)
            .arg(&input)
            .arg(&output)
            .arg_local::<T>(WG_SIZE)
            .build()?;

        unsafe { kernel.enq()? }

        buffer = output;
    }

    let mut result = vec![id; buffer.len()];
    buffer.read(&mut result).enq()?;
    Ok(result)
}
