def run_forward():
    from megengine.core._imperative_rt.core2 import apply
    from megengine.core.ops import custom
    from megengine.utils import custom_op_tools

    lib_path = custom_op_tools.build_and_load(
        "matmul_scale",
        ["src/matmul_scale.cu"],
        build_dir="build"
    )

    def matmul_scale(lhs, rhs, scale):
        op = custom.MatMulScaleForward(scale=scale)
        return apply(op, lhs, rhs)

    import numpy as np
    from megengine.tensor import Tensor

    lhs = Tensor(np.random.randn(2, 4))
    rhs = Tensor(np.random.randn(4, 8))
    res = matmul_scale(lhs, rhs, 0.1)

def run_forward_and_backward():        
    import numpy as np
    from megengine.autodiff import Function, GradManager
    from megengine.core._imperative_rt.core2 import apply
    from megengine.core.ops import custom
    from megengine.tensor import Tensor
    from megengine.utils import custom_op_tools

    lib_path = custom_op_tools.build_and_load(
        "matmul_scale",
        ["src/matmul_scale.cu"],
        build_dir="build"
    )

    class MatMulScaleFunc(Function):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def forward(self, lhs, rhs):
            self.lhs = lhs
            self.rhs = rhs
            op = custom.MatMulScaleForward(scale=self.scale)
            return apply(op, lhs, rhs)

        def backward(self, ograd):
            op = custom.MatMulScaleBackward(scale=self.scale)
            return apply(op, ograd, self.lhs, self.rhs)

    matmul = MatMulScaleFunc(scale=0.1)
    lhs = Tensor(np.random.randn(2, 4))
    rhs = Tensor(np.random.randn(4, 8))
    grad = Tensor(np.random.randn(2, 8))

    with GradManager().attach([lhs, rhs]) as gm:
        (y,) = matmul(lhs, rhs)
        gm.backward(y)

run_only_forward = False

if __name__ == '__main__':
    if run_only_forward:
        run_forward()
    else:
        run_forward_and_backward()
