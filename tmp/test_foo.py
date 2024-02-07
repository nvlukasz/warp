import warp as wp
from warp.foo import Foo


# Kernel that accesses Foo variables by name
@wp.kernel
def foo_vars_kernel(foo_id: wp.uint64, a: wp.array(dtype=float)):
    i = wp.tid()

    foo = wp.foo_get(foo_id)

    a[i] = foo.magic + float(foo.offset)


# Kernel that demonstrates 1d indexing of Foo
@wp.kernel
def foo_index_1d_kernel(foo_id: wp.uint64, a: wp.array(dtype=float)):
    i = wp.tid()

    foo = wp.foo_get(foo_id)

    a[i] = foo[i]


# Kernel that demonstrates 2d indexing of Foo
@wp.kernel
def foo_index_2d_kernel(foo_id: wp.uint64, a: wp.array2d(dtype=float)):
    i, j = wp.tid()

    foo = wp.foo_get(foo_id)

    a[i, j] = foo[i, j]


def test_foo_vars():
    a = wp.zeros(10, dtype=float)

    foo = Foo(magic=0.5, offset=3)

    wp.launch(foo_vars_kernel, dim=a.size, inputs=[foo.id, a])
    print(a)


def test_foo_index_1d():
    a = wp.zeros(10, dtype=float)

    foo = Foo(magic=0.1)

    wp.launch(foo_index_1d_kernel, dim=a.size, inputs=[foo.id, a])
    print(a)


def test_foo_index_2d():
    a = wp.zeros((10, 10), dtype=float)

    foo = Foo(offset=5)

    wp.launch(foo_index_2d_kernel, dim=a.shape, inputs=[foo.id, a])
    print(a)


def run_tests(device):
    print(f"===== Device {device} ====================")
    with wp.ScopedDevice(device):
        print("Test vars:")
        test_foo_vars()

        print("Test 1d indexing:")
        test_foo_index_1d()

        print("Test 2d indexing:")
        test_foo_index_2d()


wp.init()

# it's useful to clear the kernel cache when developing new codegen features
wp.build.clear_kernel_cache()

wp.force_load(device=["cpu", "cuda:0"])

run_tests("cpu")
run_tests("cuda:0")
