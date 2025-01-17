from __future__ import annotations
from . import helpers
from .plot_file import plot_file  # noqa
from .definitons import Definitions
from .block import parse_blocks, write_blocks, Block
from .rf import parse_rfs, write_rfs, Rf  # noqa
from .trap import parse_traps, write_traps, Trap
from .gradient import parse_gradients, write_grads, Gradient
from .adc import parse_adcs, write_adcs, Adc  # noqa

# Supports version 1.2.0 to 1.4.0, python representation is modeled after 1.4.0


class PulseqFile:
    def __init__(self, file_name: str) -> None:
        sections = helpers.file_to_sections(file_name)

        assert "VERSION" in sections
        self.version = helpers.parse_version(sections.pop("VERSION"))
        assert 120 <= self.version <= 140

        # mandatory sections
        assert "BLOCKS" in sections
        assert self.version != 140 or "DEFINITIONS" in sections
        assert not (self.version == 140 and "DELAYS" in sections)

        if "DEFINITIONS" in sections:
            self.definitions = Definitions.parse(
                sections.pop("DEFINITIONS"), self.version)
        else:
            self.definitions = Definitions()

        # Parse [RF], [GRADIENTS], [TRAP], [ADC], [SHAPES]
        # They are dicts of (ID, event) so return an empty dict if not present
        def maybe_parse(sec_name, parser):
            if sec_name not in sections:
                return {}
            else:
                return parser(sections.pop(sec_name), self.version)

        self.rfs = maybe_parse("RF", parse_rfs)
        self.grads = helpers.merge_dicts(
            maybe_parse("GRADIENTS", parse_gradients),
            maybe_parse("TRAP", parse_traps),
        )
        self.adcs = maybe_parse("ADC", parse_adcs)
        self.shapes = maybe_parse("SHAPES", helpers.parse_shapes)

        # Finally parse the blocks, some additional logic is needed to convert
        # 1.3.x sequences with delay events into the 1.4.0 format
        if self.version == 140:
            self.blocks = parse_blocks(
                sections.pop("BLOCKS"), self.version,
                None, self.definitions.block_raster_time
            )
        else:
            delays = maybe_parse("DELAYS", helpers.parse_delays)
            self.blocks = parse_blocks(
                sections.pop("BLOCKS"), self.version,
                delays, None
            )

        # Inform if there are sections that were not parsed
        if len(sections) > 0:
            print(f"Some sections were ignored: {list(sections.keys())}")

        # Calculate block durations for 1.3.x sequences
        def calc_duration(block: Block) -> float:
            durs = [block.duration]  # delay event for 1.3.x

            if block.adc_id != 0:
                durs.append(self.adcs[block.adc_id].get_duration())

            if block.rf_id != 0:
                durs.append(self.rfs[block.rf_id].get_duration(
                    self.definitions.rf_raster_time, self.shapes
                ))

            grads = [
                self.grads.get(block.gx_id, None),
                self.grads.get(block.gy_id, None),
                self.grads.get(block.gz_id, None)
            ]

            for grad in grads:
                if isinstance(grad, Gradient):
                    durs.append(grad.get_duration(
                        self.definitions.grad_raster_time, self.shapes
                    ))
                if isinstance(grad, Trap):
                    durs.append(grad.get_duration())

            return max(durs)

        for block in self.blocks.keys():
            # We could check if 1.4.0 has set correct durations
            self.blocks[block].duration = calc_duration(self.blocks[block])

    def save(self, file_name: str):
        with open(file_name, "w") as out:
            out.write(
                "# Pulseq sequence definition file\n"
                "# Re-Exported by the MRzero pulseq interpreter\n"
            )
            helpers.write_version(out, 140)
            self.definitions.write(out)
            write_blocks(out, self.blocks, self.definitions.block_raster_time)
            write_rfs(out, self.rfs)
            write_traps(
                out,
                {k: v for k, v in self.grads.items() if isinstance(v, Trap)}
            )
            write_grads(
                out,
                {k: v for k, v in self.grads.items()
                    if isinstance(v, Gradient)}
            )
            write_adcs(out, self.adcs)
            helpers.write_shapes(out, self.shapes)

    def __repr__(self) -> str:
        return (
            f"PulseqFile(version={self.version}, "
            f"definitions={self.definitions}, "
            f"blocks={self.blocks}, "
            f"rfs={self.rfs}, "
            f"adcs={self.adcs}, "
            f"grads={self.grads}, "
            f"shapes={self.shapes})"
        )
