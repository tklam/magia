import inspect
import logging
from collections import Counter, OrderedDict
from dataclasses import dataclass
from functools import cached_property
from itertools import count
from os import PathLike
from pathlib import Path
from string import Template
from typing import Optional, Union, List, Set, Self, cast
from enum import Enum
from dataclasses import dataclass, field

from .core import Signal, SignalDict, SignalType, Synthesizable, Case, Constant

logger = logging.getLogger(__name__)


class VarValue(Enum):
    ZERO = 0
    ONE = 1
    DONT_CARE = 2

    @classmethod
    def unsigned_int_to_var_values(cls, unsigned_int: int, num_vars: int) -> List[Self]:
        """
        Convert an unsigned int to a list of variable values:
        - The 0th element corresponds to the rightmost bit
        - The last element corresponds to the leftmost bit
        E.g.
        index    0 1 2 3 4 5 6 7
        -------------------------
        0x04 => [0,0,1,0,0,0,0,0]
        """
        var_values = []
        for i in range(0, num_vars):
            if unsigned_int & (1 << i) != 0:
                var_values.append(VarValue.ONE)
            else:
                var_values.append(VarValue.ZERO)
        return var_values

    @classmethod
    def dont_care_var_values(cls, num_vars: int) -> List[Self]:
        """
        return a list of `num_vars` don't cares
        """
        return [VarValue.DONT_CARE for i in range(0, num_vars)]


class Implicant:
    def __init__(
        self,
        num_vars: int,
        var_values: List[VarValue],
        is_dont_care: bool,
        covers: List[Self] = None,
    ):
        self._num_vars = num_vars  # number of variables/bits
        self._var_values = var_values
        self._num_ones = self.count_num_ones(self._var_values)
        self._is_dont_care = is_dont_care
        self._covers = covers  # the implicants implying this implicant
        if self._covers is not None:
            for c in self._covers:
                c.add_covered_by(self)
            self._num_dont_care_bits = self._covers[0].num_dont_care_bits + 1
        else:
            self._num_dont_care_bits = len(
                [1 for v in var_values if v == VarValue.DONT_CARE]
            )
        self._covered_by = set({})  # the implicants that cover this implicant

    def count_num_ones(self, var_values: List[VarValue]) -> int:
        return len([1 for v in var_values if v == VarValue.ONE])

    def set_var_values_by_unsigned_int(self, unsigned_int: int):
        """
        Given an unsigned int `unsigned_int`, set:
        - `self._var_values`
        - `self._num_ones`
        - `self._num_dont_care_bits`
        """
        self._var_values.clear()
        self._var_values = VarValue.unsigned_int_to_var_values(
            unsigned_int, self._num_vars
        )
        self._num_ones = self.count_num_ones(self._var_values)
        self._num_dont_care_bits = len(
            [1 for v in var_values if v == VarValue.DONT_CARE]
        )

    @property
    def var_values(self) -> Set[int]:
        return self._var_values

    @property
    def is_dont_care(self) -> bool:
        return self._is_dont_care

    @property
    def num_ones(self) -> int:
        return self._num_ones

    @property
    def covers(self) -> List[Self]:
        return self._covers

    @property
    def covered_by(self) -> List[Self]:
        return self._covered_by

    @property
    def num_dont_care_bits(self) -> int:
        return self._num_dont_care_bits

    def add_covered_by(self, imp: Self):
        self._covered_by.add(imp)

    def is_mergable_with(self, another: Self) -> bool:
        """
        Return True if `another` can be merged with this implicant; False otherwise
        """
        num_diff = 0
        for i in range(0, self._num_vars):
            if self._var_values[i] != another.var_values[i]:
                num_diff += 1
            if num_diff >= 2:
                return False
        return True

    def merge(self, another: Self) -> Self:
        """
        Merge with `another` to create a new Implicant
        """
        if not self.is_mergable_with(another):
            return None

        new_var_values = self._var_values.copy()
        for i in range(0, self._num_vars):
            if self._var_values[i] != another.var_values[i]:
                new_var_values[i] = VarValue.DONT_CARE

        return Implicant(self._num_vars, new_var_values, False, [self, another])

    def get_terminal_implicants(self) -> List[Self]:
        """
        Return a list of terminal Implicants that cannot further cover other implicants (minterms)
        """
        cover_stack = [self]
        implied_imps = []
        while len(cover_stack) > 0:
            imp = cover_stack.pop()
            if imp.covers is None:
                implied_imps.append(imp)
            else:
                for p in imp.covers:
                    cover_stack.append(p)
        return implied_imps

    def __str__(self) -> str:
        return self.var_values_to_str(self._var_values)

    @classmethod
    def var_values_to_str(cls, var_values: List[VarValue]) -> str:
        s = ""
        for i in range(len(var_values) - 1, -1, -1):
            match (var_values[i]):
                case VarValue.ONE:
                    s += "1"
                case VarValue.ZERO:
                    s += "0"
                case VarValue.DONT_CARE:
                    s += "-"
        return s

    @classmethod
    def var_values_to_int(cls, var_values: List[VarValue]) -> int:
        num = 0
        for i in range(0, len(var_values)):
            match (var_values[i]):
                case VarValue.ONE:
                    num += 1 << i
                case VarValue.DONT_CARE:
                    raise NotImplementedError
        return num


class SingleOutputCaseOptimizerQM(Case.CaseOptimizer):
    """
    SingleOutputCaseOptimizerQM can optimize non-unique Case statements using
    Quine McCluskey method. If the output is a multibit bus, every bit will be
    optimized individually.
    """

    @dataclass
    class Problem:
        output_bus_name: str
        output_index: int
        _output_name: str = None
        _implicants: List[Implicant] = field(default_factory=list)
        _essential_prime_implicants: List[Implicant] = field(default_factory=list)

        @property
        def output_name(self) -> str:
            if self._output_name is None:
                self._output_name = f"{self.output_bus_name}[{str(self.output_index)}]"
            return self._output_name

        @property
        def implicants(self) -> List[Implicant]:
            return self._implicants

        @property
        def essential_prime_implicants(self) -> List[Implicant]:
            return self._essential_prime_implicants

        def add_implicant(self, imp: Implicant) -> None:
            self._implicants.append(imp)

        def set_essential(self, imp: Implicant):
            self._essential_prime_implicants.append(imp)

        def __str__(self) -> str:
            s = f"{self.output_name}:\n"
            for i in self._implicants:
                s += f"    {str(i)}"
                if not i.is_dont_care:
                    s += " 1"
                else:
                    s += " -"
                s += "\n"
            return s

    def __init__(self):
        pass

    def optimize_problem(
        self, num_support_vars: int, initial_problem: Problem
    ) -> Problem:
        problem_stack = [initial_problem]
        level = 0
        prime_problem = None

        # 1. Find the prime implicants
        while len(problem_stack) > 0:
            problem = problem_stack.pop()

            num_ones_mask = [False for i in range(0, num_support_vars + 1)]
            implicant_groups: dict[int, List[Implicant]] = {}
            for i in problem.implicants:
                num_ones_mask[i.num_ones] = True

                if i.num_ones in implicant_groups:
                    implicant_groups[i.num_ones].append(i)
                else:
                    implicant_groups[i.num_ones] = [i]

            merged_implicants = []
            for i in range(0, num_support_vars):
                if num_ones_mask[i] == False or num_ones_mask[i + 1] == False:
                    continue
                for x in implicant_groups[i]:
                    for y in implicant_groups[i + 1]:
                        if not x.is_mergable_with(y):
                            continue
                        merged_implicant = x.merge(y)
                        if merged_implicant is None:
                            continue
                        merged_implicants.append(merged_implicant)

            if len(merged_implicants) > 0:
                for i in merged_implicants:
                    implicant_groups[i.num_ones].append(i)

                new_problem = SingleOutputCaseOptimizerQM.Problem(
                    problem.output_bus_name, problem.output_index
                )
                implicants_str_cache = set({})
                for i, implicants in implicant_groups.items():
                    for imp in implicants:
                        if len(imp.covered_by) > 0:
                            # those who have already been merged
                            continue
                        if str(imp) in implicants_str_cache:
                            continue
                        new_problem.add_implicant(imp)
                        implicants_str_cache.add(str(imp))

                logger.debug(
                    f"------------------------------ Merging implicants ({level}):"
                )
                logger.debug("Current:")
                logger.debug(problem)
                logger.debug("")
                logger.debug("Next:")
                logger.debug(new_problem)

                problem_stack.append(new_problem)

                level += 1
            else:
                # all the implicants are prime implicants
                prime_problem = problem
                break

        # 2. Find the essential prime implicants
        essential_minterms = set({})
        for i in initial_problem.implicants:
            if i.is_dont_care == False:
                essential_minterms.add(i)

        if len(essential_minterms) > 0:
            for i in sorted(
                prime_problem.implicants,
                key=lambda x: x.num_dont_care_bits,
                reverse=True,
            ):
                for c in i.get_terminal_implicants():
                    if c not in essential_minterms:
                        continue
                    prime_problem.set_essential(i)
                    essential_minterms.remove(c)
                    break
        else:
            # Only off-set is defined => treat all don't cares as potential on-set
            prime_problem.set_essential(
                sorted(
                    prime_problem.implicants,
                    key=lambda x: x.num_dont_care_bits,
                    reverse=True,
                )[0]
            )

        return prime_problem

    def optimize(self, case_statement: Case) -> dict[Union[int], Union[Signal, int]]:
        """
        Optimize for each output
        """
        #
        # TODO
        #   1. currently optimization works only if all the drivers are `int`
        #   2. consider the complement of the problem to see whether there will be fewer implicants and they are smaller
        #   3. consider all problems at the same time (perhaps, as a new optimizer)
        if any(
            [isinstance(driver, Signal) for driver in case_statement.cases.values()]
        ):
            raise NotImplementedError

        DRIVERS_SELECTOR_INDEX = 0

        # 1. collect output variables
        num_output_vars = case_statement.output_width
        output_vars_bundle_name = case_statement.net_name
        logger.debug(f"Number of output variables: {num_output_vars}")

        # 2. collect input variables and minterms
        # support variables = {selector variables} + {driver variables = empty}
        # #TODO support variables = {selector variables} + {driver variables}
        case_selector = case_statement.drivers[DRIVERS_SELECTOR_INDEX]
        selector_vars = [
            case_selector[i] for i in range(case_selector.width - 1, -1, -1)
        ]
        driver_vars = []
        support_vars = selector_vars + driver_vars
        num_support_vars = len(support_vars)
        num_minterms = 2**num_support_vars

        for s in support_vars:
            logger.debug(f"Support variable: {s.name}")

        # 3. create data structures for every output

        # problems are ordered from the rightmost to the leftmost bit
        problems = [
            SingleOutputCaseOptimizerQM.Problem(output_vars_bundle_name, i)
            for i in range(0, num_output_vars)
        ]
        for i in range(0, num_minterms):
            # i is the selector value
            selector_var_values = VarValue.unsigned_int_to_var_values(
                i, num_support_vars
            )
            if i in case_statement.cases.keys():
                # `case_statement` has defined this minterm
                driver_var_values = VarValue.unsigned_int_to_var_values(
                    case_statement.cases[i], num_support_vars
                )
            else:
                driver_var_values = VarValue.dont_care_var_values(num_support_vars)

            for j in range(0, num_output_vars):
                match driver_var_values[j]:
                    case VarValue.ONE:
                        problems[j].add_implicant(
                            Implicant(num_support_vars, selector_var_values, False)
                        )
                    case VarValue.DONT_CARE:
                        problems[j].add_implicant(
                            Implicant(num_support_vars, selector_var_values, True)
                        )

        # 4. solve all problems and cache the covered minterms
        problem_to_covered_minterms: dict[int, List[int]] = (
            {}
        )  # memory address of problem -> its minterms id
        optimized_case_table = {}
        for p in problems:
            logger.debug(f"------ Optimizing problem:\n {p}")
            problem_solution = self.optimize_problem(num_support_vars, p)
            logger.debug(
                "-------------------------------------------- Final solution to the problem:"
            )
            logger.debug(problem_solution)
            solution_implicants = problem_solution.essential_prime_implicants
            problem_to_covered_minterms[id(p)] = []
            for i in solution_implicants:
                logger.debug(f"  essential implicant: {i}")
                for c in i.get_terminal_implicants():
                    problem_to_covered_minterms[id(p)].append(
                        Implicant.var_values_to_int(c.var_values)
                    )
                    logger.debug(
                        f"      covered minterm: {c} [don't care? {c.is_dont_care}]"
                    )
            logger.debug("--------------------------------------------")

        # 5. construct the combined case table for all problems from the leftmost to the rightmost bit
        logger.debug("---------------Constructing the new case table")
        for i in range(0, num_minterms):
            driver_value = 0
            bit_shift = 0
            for p in problems:
                covered_minterm_ids = set(sorted(problem_to_covered_minterms[id(p)]))
                if i in covered_minterm_ids:
                    driver_value |= 1 << bit_shift
                bit_shift += 1
            optimized_case_table[i] = driver_value
            logger.debug(
                f"  {i:0{num_support_vars}b}: {driver_value:0{len(problems)}b}"
            )

        return optimized_case_table
