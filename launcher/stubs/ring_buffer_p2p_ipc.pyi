"""Type stubs for the ring_buffer_p2p_ipc pybind11 extension."""

def fill_source(ptr: int, count: int, seed: int) -> None: ...
def run_sender(
    src_buf: int,
    recv_buf: int,
    tail: int,
    head: int,
    slot_count: int,
    num_slots: int,
    total_slots: int,
) -> None: ...
def run_receiver(
    dst_buf: int,
    recv_buf: int,
    tail: int,
    head: int,
    slot_count: int,
    num_slots: int,
    total_slots: int,
) -> None: ...
def verify(ptr: int, count: int, seed: int) -> bool: ...
