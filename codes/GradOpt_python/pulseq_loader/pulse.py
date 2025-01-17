from __future__ import annotations
import numpy as np
from .pulseq_file import PulseqFile, Block
from .helpers import split_gradm
from .spoiler import Spoiler


class Pulse:
    def __init__(
        self,
        angle: float,
        phase: float,
    ) -> None:
        self.angle = angle
        self.phase = phase

    @classmethod
    def parse(cls, block: Block, pulseq: PulseqFile
              ) -> tuple[Spoiler, Pulse, Spoiler]:
        rf = pulseq.rfs[block.rf_id]
        raster_time = pulseq.definitions.rf_raster_time

        if rf.time_id != 0:
            time = pulseq.shapes[rf.time_id]
        else:
            time = np.arange(len(pulseq.shapes[rf.mag_id]))
        event_time = np.concatenate([time[1:] - time[:-1], [1]]) * raster_time

        mag = 2*np.pi * rf.amp * pulseq.shapes[rf.mag_id]
        phase = pulseq.shapes[rf.phase_id]
        pulse = mag * event_time * np.exp(2j*np.pi * phase)

        # Pulses with complex phases are not supported
        assert np.sum(np.abs(pulse.imag)) < np.pi/180  # Trigger at 1°
        pulse = pulse.real

        center = np.argmax(np.cumsum(pulse) > np.sum(pulse) / 2)
        t = float(rf.delay + center * raster_time)

        gradm = np.zeros((2, 3))
        if block.gx_id != 0:
            gradm[:, 0] = split_gradm(pulseq.grads[block.gx_id], pulseq, t)
        if block.gy_id != 0:
            gradm[:, 1] = split_gradm(pulseq.grads[block.gy_id], pulseq, t)
        if block.gz_id != 0:
            gradm[:, 2] = split_gradm(pulseq.grads[block.gz_id], pulseq, t)

        fov = pulseq.definitions.fov
        gradm[:, 0] *= fov[0]
        gradm[:, 1] *= fov[1]
        gradm[:, 2] *= fov[2]

        return (
            Spoiler(t, gradm[0, :]),
            cls(np.sum(pulse), rf.phase),
            Spoiler(block.duration - t, gradm[1, :])
        )

    def __repr__(self) -> str:
        return (
            f"Pulse(angle={self.angle*180/np.pi:.1f}°, "
            f"phase={self.phase*180/np.pi:.1f}°)"
        )
