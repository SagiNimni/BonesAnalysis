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
        directory = 'manipulations/ParralerMethods/'
        if mute:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        try:
            f = open(directory + filename, 'r')
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
        self.original_buff = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(self.image_array))
        self.dst_buff = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.image_array.nbytes)

    def execute(self, method, *args):
        try:
            if method == "GradientCalculation":
                mf = cl.mem_flags
                self.angle_buff = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.image_array.astype(np.double).nbytes)
                mask_buff = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=args[0].astype(np.uint8))
                self.program.GradientCalculation(self.queue, self.image_array.shape, None, self.original_buff,
                                                 self.dst_buff, self.angle_buff, mask_buff, np.uint32(args[1]),
                                                 np.float32(args[2]), np.float32(args[3]), np.float32(args[4]))
                image = np.empty_like(self.image_array)
                angle = np.empty_like(self.image_array.astype(np.double))
                cl._enqueue_read_buffer(self.queue, self.dst_buff, image)
                cl._enqueue_read_buffer(self.queue, self.angle_buff, angle)

                mask_buff.release()
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

            elif method == "removeSmallEdges":
                mf = cl.mem_flags
                self.label_buff = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array(list(args[0].keys())).astype(np.uint32))
                self.count_buff = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array(list(args[0].values())).astype(np.uint32))
                self.program.removeSmallEdges(self.queue, self.image_array.shape, None, self.original_buff,
                                              self.dst_buff, self.label_buff, self.count_buff,
                                              np.uint(args[1]), np.uint(len(list(args[0].keys()))))
                result = np.empty_like(self.image_array)
                cl._enqueue_read_buffer(self.queue, self.dst_buff, result)

                self.original_buff = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=result)
                self.dst_buff = cl.Buffer(self.ctx, mf.WRITE_ONLY, (self.image_array.astype(np.uint8)).nbytes)
                self.program.ConvertLabelsToEdges(self.queue, self.image_array.shape, None, self.original_buff,
                                                  self.dst_buff)
                edges = np.empty_like(self.image_array).astype(np.uint8)
                cl._enqueue_read_buffer(self.queue, self.dst_buff, edges)

                self.label_buff.release()
                self.count_buff.release()
                self.original_buff.release()
                self.dst_buff.release()

                return result, edges

            elif method == "makeBackground":
                a = np.uint8(args[0])
                b = np.uint8(args[1])
                self.program.makeBackground(self.queue, self.image_array.shape, None, self.original_buff,
                                            self.dst_buff, a[0], a[1], a[2], b[0], b[1], b[2])
                result = np.empty_like(np.ascontiguousarray(self.image_array))
                cl._enqueue_read_buffer(self.queue, self.dst_buff, result)

                self.original_buff.release()
                self.dst_buff.release()

                return result

            elif method == "removeShapesInsideShape":
                self.program.removeShapesInsideShape(self.queue, self.image_array.shape, None, self.original_buff,
                                                     self.dst_buff, np.uint32(args[0]))
                result = np.empty_like(self.image_array)
                cl._enqueue_read_buffer(self.queue, self.dst_buff, result)

                self.original_buff.release()
                self.dst_buff.release()

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
