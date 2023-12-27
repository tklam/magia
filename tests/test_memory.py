from pathlib import Path

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge
from cocotb_test.simulator import run as sim_run

from magia import Input, Memory, Module, Output


async def drive_spram(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await FallingEdge(dut.clk)  # Synchronize with the clock

    dut.wen.value = 1
    dut.addr.value = 0x12
    dut.din.value = 0xAB
    await FallingEdge(dut.clk)
    dut.din.value = 0xCD
    await FallingEdge(dut.clk)


@cocotb.test()
async def spram_write_through(dut):
    """
    Write through shall return the value of the intermediate write
    """
    await drive_spram(dut)
    assert dut.dout.value == 0xCD, "Failure on write through"


@cocotb.test()
async def spram_read_first(dut):
    """
    Read first shall return the value of the previous write
    """
    await drive_spram(dut)
    assert dut.dout.value == 0xAB, "Failure on write through"


class TestMemory:
    TOP = "TopModule"

    class SPRAM(Module):
        def __init__(self, rw_write_through, **kwargs):
            super().__init__(**kwargs)

            self.io += [
                Input("clk", 1),
                Input("wen", 1),
                Input("addr", 8),
                Input("din", 8),
                Output("dout", 8),
            ]

            mem = Memory.SP(
                self.io.clk, 8, 8,
                rw_write_through=rw_write_through,
            )

            for port in ["addr", "din", "wen"]:
                mem.rw_port()[port] <<= self.io[port]
            mem.rw_port().en <<= 1
            self.io.dout <<= mem.rw_port().dout

    def test_sp_write_through(self, temp_build_dir):
        ram = self.SPRAM(rw_write_through=True, name=self.TOP)
        with pytest.elaborate_to_file(ram) as filename:
            sim_run(
                simulator="verilator",  # simulator
                verilog_sources=[filename],  # sources
                toplevel=self.TOP,  # top level HDL
                python_search=[str(Path(__name__).parent.absolute())],  # python search path
                module=Path(__name__).name,  # name of cocotb test module
                testcase="spram_write_through",  # name of test function
                sim_build=temp_build_dir,  # temp build directory
                work_dir=temp_build_dir,  # simulation  directory
            )

    def test_sp_read_first(self, temp_build_dir):
        ram = self.SPRAM(rw_write_through=False, name=self.TOP)
        with pytest.elaborate_to_file(ram) as filename:
            sim_run(
                simulator="verilator",  # simulator
                verilog_sources=[filename],  # sources
                toplevel=self.TOP,  # top level HDL
                python_search=[str(Path(__name__).parent.absolute())],  # python search path
                module=Path(__name__).name,  # name of cocotb test module
                testcase="spram_read_first",  # name of test function
                sim_build=temp_build_dir,  # temp build directory
                work_dir=temp_build_dir,  # simulation  directory
            )
