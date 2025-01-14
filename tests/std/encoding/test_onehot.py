from pathlib import Path

import cocotb
import pytest
from cocotb.triggers import Timer
from cocotb_test.simulator import run as sim_run

import tests.helper as helper
from magia import Input, Module, Output
from magia.std.encoding import binary_to_onehot, onehot_to_binary

cocotb_test_prefix = "coco_"


@cocotb.test()
async def onehot_with_width(dut, width):
    assert len(dut.onehot) == 2 ** width, f"Expected {2 ** width}, got {len(dut.onehot)}"
    for i in range(2 ** width):
        dut.binary_in.value = i
        await Timer(1, units="ns")
        assert dut.onehot.value == 2 ** i, f"Expected {2 ** i}, got {dut.onehot.value.integer}"
        assert dut.binary.value == i, f"Expected {i}, got {dut.binary.value.integer}"


test_gen, onehot_width_params, onehot_width_values = helper.parameterized_testbench(
    onehot_with_width, [(i,) for i in [2, 3, 4, 5, 8]],
)
test_gen()


@cocotb.test()
async def onehot_with_max_value(dut, max_value):
    assert len(dut.onehot) == max_value + 1, f"Expected {max_value}, got {len(dut.onehot)}"
    for i in range(max_value + 1):
        dut.binary_in.value = i
        await Timer(1, units="ns")
        assert dut.onehot.value == 2 ** i, f"Expected {2 ** i}, got {dut.onehot.value.integer}"
        assert dut.binary.value == i, f"Expected {i}, got {dut.binary.value.integer}"


test_gen, onehot_max_params, onehot_max_values = helper.parameterized_testbench(
    onehot_with_max_value, [(i,) for i in [1, 2, 3, 4, 5, 7, 9, 13, 17, 101]],
)
test_gen()


class TestOneHot:
    TOP = "TopModule"

    class OneHotLoop(Module):
        def __init__(self, width, **kwargs):
            super().__init__(**kwargs)

            self.io += Input("binary_in", width)
            self.io += Output("onehot", 2 ** width)
            self.io += Output("binary", width)

            onehot = binary_to_onehot(self.io.binary_in)
            self.io.onehot <<= onehot
            self.io.binary <<= onehot_to_binary(onehot)

    class OneHotLoopWithMax(Module):
        def __init__(self, max_value, **kwargs):
            super().__init__(**kwargs)

            self.io += Input("binary_in", max_value.bit_length())
            self.io += Output("onehot", max_value + 1)
            self.io += Output("binary", max_value.bit_length())

            onehot = binary_to_onehot(self.io.binary_in, max_value)
            self.io.onehot <<= onehot
            self.io.binary <<= onehot_to_binary(onehot)

    @pytest.mark.parametrize(onehot_width_params, onehot_width_values)
    def test_width(self, width, cocotb_testcase, temp_build_dir):
        with helper.elaborate_to_file(
                self.OneHotLoop(width=width, name=self.TOP)
        ) as filename:
            sim_run(
                simulator="verilator",  # simulator
                verilog_sources=[filename],  # sources
                toplevel=self.TOP,  # top level HDL
                python_search=[str(Path(__name__).parent.absolute())],  # python search path
                module=Path(__name__).name,  # name of cocotb test module
                testcase=cocotb_testcase,  # name of test function
                sim_build=temp_build_dir,  # temp build directory
                work_dir=temp_build_dir,  # simulation  directory
            )

    @pytest.mark.parametrize(onehot_max_params, onehot_max_values)
    def test_max_value(self, max_value, cocotb_testcase, temp_build_dir):
        with helper.elaborate_to_file(
                self.OneHotLoopWithMax(max_value, name=self.TOP)
        ) as filename:
            sim_run(
                simulator="verilator",  # simulator
                verilog_sources=[filename],  # sources
                toplevel=self.TOP,  # top level HDL
                python_search=[str(Path(__name__).parent.absolute())],  # python search path
                module=Path(__name__).name,  # name of cocotb test module
                testcase=cocotb_testcase,  # name of test function
                sim_build=temp_build_dir,  # temp build directory
                work_dir=temp_build_dir,  # simulation  directory
            )
