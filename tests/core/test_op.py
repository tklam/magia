import random
from pathlib import Path

import cocotb
import cocotb.clock
import pytest
from cocotb.regression import TestFactory
from cocotb_test.simulator import run as sim_run

from magia import Input, Module, Output

cocotb_test_prefix = "coco_"


@cocotb.test()
async def when_as_mux_test(dut):
    """ Test if the `when` operator works as a mux """
    for _ in range(50):
        a = random.randint(0, 0xFF)
        b = random.randint(0, 0xFF)
        sel = random.randint(0, 1)

        dut.a.value = a
        dut.b.value = b
        dut.sel.value = sel

        await cocotb.clock.Timer(1, units="ns")

        if sel == 1:
            assert dut.q.value == a
        else:
            assert dut.q.value == b


@cocotb.test()
async def when_as_mux_comp(dut):
    """ Test if the `when` operator works as a Comparator """
    for _ in range(16 * 16):
        a = random.randint(0, 0xF)
        b = random.randint(0, 0xF)

        dut.a.value = a
        dut.b.value = b

        await cocotb.clock.Timer(1, units="ns")

        if a != b:
            assert dut.q.value == a
        else:
            assert dut.q.value == 0xF


@cocotb.test()
async def case_as_mux(dut, selector, selection: dict[int, int]):
    """ Test if the `case` operator works as a mux """
    for _ in range(50):

        sel = random.randint(0, 2**selector - 1)
        dut.sel.value = sel
        for i in range(2**selector):
            getattr(dut, f"d_{i}").value = random.randint(1, 0xFF)

        await cocotb.clock.Timer(1, units="ns")

        if sel in selection:
            assert dut.q.value == getattr(dut, f"d_{selection[sel]}").value
        else:
            for i in range(2 ** selector):
                assert dut.q.value != getattr(dut, f"d_{i}").value


case_as_mux_test_opts = ["selector", "selection"]
case_as_mux_test_opts_val: list = [
    (1, {1: 0}),
    (1, {0: 0, 1: 1}),
    (2, {0: 0, 3: 2}),
    (2, {0: 0, 1: 1, 3: 2}),
    (2, {0: 0, 1: 1, 2: 2, 3: 3}),
    (3, {0: 0, 4: 1}),
    (3, {i: i for i in range(8)}),
]
case_as_mux_pytest_param = ",".join(case_as_mux_test_opts + ["cocotb_testcase"])
case_as_mux_pytest_param_val = [
    val + (f"{cocotb_test_prefix}case_as_mux_{i + 1:03d}",)
    for i, val in enumerate(case_as_mux_test_opts_val)
]
tf_reg_test = TestFactory(test_function=case_as_mux)
tf_reg_test.add_option(case_as_mux_test_opts, case_as_mux_test_opts_val)
tf_reg_test.generate_tests(prefix=cocotb_test_prefix)


class TestOperations:
    TOP = "TopModule"

    def test_when_mux(self, temp_build_dir):
        class SimpleMux(Module):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

                self.io += Input("a", 8)
                self.io += Input("b", 8)
                self.io += Input("sel", 1)
                self.io += Output("q", 8)

                self.io.q <<= self.io.a.when(self.io.sel, else_=self.io.b)

        with pytest.elaborate_to_file(
                SimpleMux(name=self.TOP)
        ) as filename:
            sim_run(
                simulator="verilator",  # simulator
                verilog_sources=[filename],  # sources
                toplevel=self.TOP,  # top level HDL
                python_search=[str(Path(__name__).parent.absolute())],  # python search path
                module=Path(__name__).name,  # name of cocotb test module
                testcase="when_as_mux_test",  # name of test function
                sim_build=temp_build_dir,  # temp build directory
                work_dir=temp_build_dir,  # simulation  directory
            )

    def test_when_cond(self, temp_build_dir):
        class Comparator(Module):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

                self.io += Input("a", 4)
                self.io += Input("b", 4)
                self.io += Output("q", 4)

                self.io.q <<= self.io.a.when(self.io.a != self.io.b, else_=0xF)

        with pytest.elaborate_to_file(
                Comparator(name=self.TOP)
        ) as filename:
            sim_run(
                simulator="verilator",  # simulator
                verilog_sources=[filename],  # sources
                toplevel=self.TOP,  # top level HDL
                python_search=[str(Path(__name__).parent.absolute())],  # python search path
                module=Path(__name__).name,  # name of cocotb test module
                testcase="when_as_mux_comp",  # name of test function
                sim_build=temp_build_dir,  # temp build directory
                work_dir=temp_build_dir,  # simulation  directory
            )

    @pytest.mark.parametrize("width, cases, expected_default", [
        (4, 4, True),
        (4, 10, True),
        (4, 16, False),
        (8, 8, True),
        (8, 16, True),
        (8, 256, False),
        (10, 16, True),
        (10, 1024, False),
    ])
    def test_case_default_unique_detection(self, width, cases, expected_default):
        class CaseModule(Module):
            def __init__(self, width, cases, **kwargs):
                super().__init__(**kwargs)

                self.io += Input("a", width)

                cases = {i: i + 1 for i in range(cases)}
                case_logic = self.io.a.case(cases)
                self.io += Output("q", len(case_logic))
                self.io.q <<= case_logic

        result = Module.elaborate_all(CaseModule(width=width, cases=cases))
        sv_code = next(iter(result.values()))
        if expected_default:
            # Expect default case to be generated
            assert "default:" in sv_code
            assert "'hX" in sv_code
        else:
            # All cases exist, no default case and unique case is apply-able
            assert "unique case" in sv_code

    @pytest.mark.parametrize(case_as_mux_pytest_param, case_as_mux_pytest_param_val)
    def test_case_as_mux(self, selector, selection, cocotb_testcase, temp_build_dir):
        class CaseMux(Module):
            def __init__(self, selector, selection, **kwargs):
                super().__init__(**kwargs)

                self.io += Input("sel", selector)
                for i in range(2 ** selector):
                    self.io += Input(f"d_{i}", 8)
                self.io += Output("q", 8)

                self.io.q <<= self.io.sel.case(cases={
                    case: self.io[f"d_{select}"]
                    for case, select in selection.items()
                }, default=None)

        with pytest.elaborate_to_file(
                CaseMux(selector, selection, name=self.TOP)
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