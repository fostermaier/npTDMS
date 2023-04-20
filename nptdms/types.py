"""Conversions to and from bytes representation of values in TDMS files"""

import datetime
import numpy as np
import numpy.typing as npt
import struct
from typing import Optional, Any, Self, Union, TYPE_CHECKING
from nptdms.timestamp import TdmsTimestamp, TimestampArray
from nptdms.log import log_manager

if TYPE_CHECKING:
    from nptdms import TdmsFile



log = log_manager.get_logger(__name__)


__all__ = [
    'numpy_data_types',
    'tds_data_types',
    'TdmsType',
    'Bytes',
    'Void',
    'Int8',
    'Int16',
    'Int32',
    'Int64',
    'Uint8',
    'Uint16',
    'Uint32',
    'Uint64',
    'SingleFloat',
    'DoubleFloat',
    'ExtendedFloat',
    'SingleFloatWithUnit',
    'DoubleFloatWithUnit',
    'ExtendedFloatWithUnit',
    'String',
    'Boolean',
    'TimeStamp',
    'ComplexSingleFloat',
    'ComplexDoubleFloat',
    'DaqMxRawData',
]


_struct_pack = struct.pack
_struct_unpack = struct.unpack


tds_data_types = {}
numpy_data_types = {}


def tds_data_type(enum_value: int, np_type: Optional[npt.DTypeLike], set_np_type: bool = True):
    def decorator(cls):
        cls.enum_value = enum_value
        cls.nptype = None if np_type is None else np.dtype(np_type)
        if enum_value is not None:
            tds_data_types[enum_value] = cls
        if set_np_type and np_type is not None:
            numpy_data_types[np.dtype(np_type)] = cls
        return cls
    return decorator


class TdmsType(object):
    size: Optional[int] = None

    def __init__(self) -> None:
        self.value: Any = None
        self.bytes: Optional[bytes] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TdmsType):
            return NotImplemented
        return self.bytes == other.bytes and self.value == other.value

    def __repr__(self) -> str:
        if self.value is None:
            return "%s" % self.__class__.__name__
        return "%s(%r)" % (self.__class__.__name__, self.value)

    @classmethod
    def read(cls, file: 'TdmsFile', endianness: str = "<") -> Any:
        raise NotImplementedError("Unsupported data type to read: %r" % cls)

    @classmethod
    def read_values(cls, file: 'TdmsFile', number_values: int, endianness: str = "<") -> Any:
        raise NotImplementedError("Unsupported data type to read: %r" % cls)


class Bytes(TdmsType):
    def __init__(self, value: bytes) -> None:
        self.value = value
        self.bytes = value


class StructType(TdmsType):
    struct_declaration: Optional[str] = None
    nptype = None

    def __init__(self, value: bytes) -> None:
        self.value = value
        self.bytes = _struct_pack('<' + self.struct_declaration, value)

    @classmethod
    def read(cls, file: 'TdmsFile', endianness: str = "<") -> Any:
        read_bytes = file.read(cls.size)
        return _struct_unpack(endianness + cls.struct_declaration, read_bytes)[0]

    @classmethod
    def from_bytes(cls, byte_array: npt.NDArray[np.byte], endianness: str = "<") -> npt.NDArray[np.byte]:
        """ Convert an array of bytes into a numpy array of data
        """
        array = byte_array.view()
        array.dtype = cls.nptype.newbyteorder(endianness)
        return array


@tds_data_type(0, None)
class Void(TdmsType):
    pass


@tds_data_type(1, np.int8)
class Int8(StructType):
    size = 1
    struct_declaration = "b"


@tds_data_type(2, np.int16)
class Int16(StructType):
    size = 2
    struct_declaration = "h"


@tds_data_type(3, np.int32)
class Int32(StructType):
    size = 4
    struct_declaration = "l"


@tds_data_type(4, np.int64)
class Int64(StructType):
    size = 8
    struct_declaration = "q"


@tds_data_type(5, np.uint8)
class Uint8(StructType):
    size = 1
    struct_declaration = "B"


@tds_data_type(6, np.uint16)
class Uint16(StructType):
    size = 2
    struct_declaration = "H"


@tds_data_type(7, np.uint32)
class Uint32(StructType):
    size = 4
    struct_declaration = "L"


@tds_data_type(8, np.uint64)
class Uint64(StructType):
    size = 8
    struct_declaration = "Q"


@tds_data_type(9, np.single)
class SingleFloat(StructType):
    size = 4
    struct_declaration = "f"


@tds_data_type(10, np.double)
class DoubleFloat(StructType):
    size = 8
    struct_declaration = "d"


@tds_data_type(11, None)
class ExtendedFloat(TdmsType):
    pass


@tds_data_type(0x19, np.single, set_np_type=False)
class SingleFloatWithUnit(StructType):
    size = 4
    struct_declaration = "f"


@tds_data_type(0x1A, np.double, set_np_type=False)
class DoubleFloatWithUnit(StructType):
    size = 8
    struct_declaration = "d"


@tds_data_type(0x1B, None)
class ExtendedFloatWithUnit(TdmsType):
    pass


@tds_data_type(0x20, None)
class String(TdmsType):
    def __init__(self, value: str) -> None:
        self.value = value
        content = value.encode('utf-8')
        length = _struct_pack('<L', len(content))
        self.bytes = length + content

    @staticmethod
    def read(file: 'TdmsFile', endianness: str = "<") -> str:
        size_bytes = file.read(4)
        size = _struct_unpack(endianness + 'L', size_bytes)[0]
        return String._decode(file.read(size))

    @classmethod
    def read_values(cls, file: 'TdmsFile', number_values: int, endianness: str = "<") -> list[str]:
        """ Read string raw data

            This is stored as an array of offsets
            followed by the contiguous string data.
        """
        offsets = [0]
        for i in range(number_values):
            offsets.append(Uint32.read(file, endianness))
        strings = []
        for i in range(number_values):
            s = file.read(offsets[i + 1] - offsets[i])
            strings.append(String._decode(s))
        return strings

    @staticmethod
    def _decode(string_bytes: bytes) -> str:
        try:
            return string_bytes.decode('utf-8')
        except UnicodeDecodeError as exc:
            log.warning(
                "Error decoding string from bytes %s, retrying with replace handler: %s",
                string_bytes, exc)
            return string_bytes.decode('utf-8', errors='replace')


@tds_data_type(0x21, np.bool_)
class Boolean(StructType):
    size = 1
    struct_declaration = "b"

    def __init__(self, value: str) -> None:
        self.value: bool = bool(value)
        self.bytes = _struct_pack('<' + self.struct_declaration, self.value)

    @classmethod
    def read(cls, file: 'TdmsFile', endianness: str = "<") -> bool:
        return bool(super(Boolean, cls).read(file, endianness))


@tds_data_type(0x44, None)
class TimeStamp(TdmsType):
    # Time stamps are stored as number of seconds since
    # 01/01/1904 00:00:00.00 UTC, ignoring leap seconds,
    # and number of 2^-64 fractions of a second.
    # Note that the TDMS epoch is not the Unix epoch.
    _tdms_epoch = np.datetime64('1904-01-01 00:00:00', 'us')
    _fractions_per_microsecond = float(10**-6) / 2**-64

    size = 16

    def __init__(self, value: Union[np.dtype[np.datetime64], datetime.datetime]) -> None:

        if not isinstance(value, np.datetime64):
            value_clean = np.datetime64(value, 'us')
        else:
            value_clean = value

        self.value = value_clean
        epoch_delta = value_clean - self._tdms_epoch

        seconds = int(epoch_delta / np.timedelta64(1, 's'))
        remainder = epoch_delta - np.timedelta64(seconds, 's')
        zero_delta = np.timedelta64(0, 's')
        if remainder < zero_delta:
            remainder = np.timedelta64(1, 's') + remainder
            seconds = seconds - 1
        microseconds = int(remainder / np.timedelta64(1, 'us'))
        second_fractions = int(microseconds * self._fractions_per_microsecond)
        self.bytes = _struct_pack('<Qq', second_fractions, seconds)

    @classmethod
    def read(cls, file: 'TdmsFile', endianness: str = "<") -> TdmsTimestamp:
        data = file.read(16)
        if endianness == "<":
            (second_fractions, seconds) = _struct_unpack(
                endianness + 'Qq', data)
        else:
            (seconds, second_fractions) = _struct_unpack(
                 endianness + 'qQ', data)
        return TdmsTimestamp(seconds, second_fractions)

    @classmethod
    def from_bytes(cls, byte_array:  npt.NDArray[np.byte], endianness: str = "<") -> TimestampArray:
        """ Convert an array of bytes to an array of timestamps
        """
        byte_array = byte_array.reshape((-1, 16))
        if endianness == "<":
            dtype = np.dtype([('second_fractions', '<u8'), ('seconds', '<i8')])
        else:
            dtype = np.dtype([('seconds', '>i8'), ('second_fractions', '>u8')])
        return TimestampArray(byte_array.view(dtype).reshape(-1))


@tds_data_type(0x08000c, np.complex64)
class ComplexSingleFloat(TdmsType):
    size = 8


@tds_data_type(0x10000d, np.complex128)
class ComplexDoubleFloat(TdmsType):
    size = 16


@tds_data_type(0xFFFFFFFF, None)
class DaqMxRawData(TdmsType):
    pass
