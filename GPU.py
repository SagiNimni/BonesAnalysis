import pyopencl as cl
import numpy as np
import sys
import os


class CL:
    """
    OpenCl class is used for running scripts on the computer GPU.
    For this we use the PyOpenCL module.
    This should be mainly used to manipulate and analyse images in the format of numpy ndarrays.

    !WARNING! The use of this class can greatly accelerate your script runtime. But it should be
    only used for scripts that are suited for the GPU architecture so use it only when needed.
    """

    def __init__(self):
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.program = None
        self.original_buff = None
        self.dst_buff = None
        self.image_array = None

    def load_program(self, filename, mute=True):
        if mute:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        try:
            f = open(filename, 'r')
            code = "".join(f.readlines())
            print(code)

            self.program = cl.Program(self.ctx, code).build()
        except (cl.Error, cl.RuntimeError, cl.LogicError, cl.MemoryError) as e:
            try:
                self.queue.finish()
                self.dst_buff.release()
                self.original_buff.release()
                self.angle_buff.release()
            except AttributeError:
                raise e
        finally:
            if mute:
                sys.stdout = old_stdout

    def load_image(self, image: np.ndarray):
        mf = cl.mem_flags
        self.image_array = image

        # Create OpenCl buffers
        self.original_buff = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.image_array)
        self.dst_buff = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.image_array.nbytes)

    def execute(self, method, *args):
        try:
            if method == "GradientCalculation":
                mf = cl.mem_flags
                self.angle_buff = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.image_array.astype(np.double).nbytes)
                self.program.GradientCalculation(self.queue, self.image_array.shape, None, self.original_buff,
                                                 self.dst_buff, self.angle_buff, np.float32(args[0]),
                                                 np.float32(args[1]), np.float32(args[2]))
                image = np.empty_like(self.image_array)
                angle = np.empty_like(self.image_array.astype(np.double))
                cl._enqueue_read_buffer(self.queue, self.dst_buff, image)
                cl._enqueue_read_buffer(self.queue, self.angle_buff, angle)

                self.angle_buff.release()
                self.dst_buff.release()
                self.original_buff.release()

                return image, angle

            elif method == "NonMaxSuppression":
                mf = cl.mem_flags
                self.angle_buff = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=args[0])
                self.program.NonMaxSuppression(self.queue, self.image_array.shape, None, self.original_buff,
                                               self.angle_buff, self.dst_buff)
                result = np.empty_like(self.image_array)
                cl._enqueue_read_buffer(self.queue, self.dst_buff, result)

                self.angle_buff.release()
                self.dst_buff.release()
                self.original_buff.release()

                return result

            elif method == "hysteresis":
                self.program.hysteresis(self.queue, self.image_array.shape, None, self.original_buff, self.dst_buff,
                                        np.uint32(args[0]), np.uint32(args[1]))
                result = np.empty_like(self.image_array)
                cl._enqueue_read_buffer(self.queue, self.dst_buff, result)

                self.dst_buff.release()
                self.original_buff.release()

                return result

            else:
                raise KeyError("The method you entered doesn't exist in the cl file")

        except (TypeError, ValueError, AttributeError) as e:
            self.queue.finish()
            raise TypeError("Image or Program wasn't loaded properly")

        except (cl.Error, cl.RuntimeError, cl.LogicError, cl.MemoryError) as e:
            self.queue.finish()
            raise e

        finally:
            self.queue.finish()
