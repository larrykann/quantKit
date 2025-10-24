"""quantKit.data.schemas

Binary struct definitions for quantKit's flat-file storage format (.qkd, .qkm, .qkt).

Each file contains a 512-byte header followed by fixed-size records. All timestamps
use Sierra Chart's DateTime format (Dec 30, 1899 epoch, microsecond precision).

See documentation for detailed format specifications and usage examples.
"""

import struct
from dataclasses import dataclass
from typing import ClassVar
from enum import Enum

__all__ = [
    "FileHeader",
    "DailyBar",
    "MinuteBar",
    "TickBar",
    "AssetType",
    "Resolution",
]


class AssetType(Enum):
    """Supported asset types.
    
    Attributes:
        EQUITY: Stocks, ETFs, equity options
        FUTURES: Futures contracts
        FOREX: Currency pairs
        CRYPTO: Cryptocurrencies
    """
    EQUITY = "equity"
    FUTURES = "futures"
    FOREX = "forex"
    CRYPTO = "crypto"


class Resolution(Enum):
    """Data resolution types.
    
    Attributes:
        DAILY: One bar per trading day
        MINUTE: One bar per minute
        TICK: Individual trades/quotes
    """
    DAILY = "daily"
    MINUTE = "minute"
    TICK = "tick"


@dataclass
class FileHeader:
    """512-byte file header containing metadata.
    
    Attributes:
        magic: Format identifier (b'QKD1', b'QKM1', or b'QKT1')
        version: Format version (currently 1)
        symbol: Trading symbol (max 31 chars)
        asset_type: Asset category (max 15 chars)
        exchange: Exchange identifier (max 31 chars)
        provider: Data source name (max 31 chars)
        bar_count: Number of records in file
        start_datetime: SC DateTime of first record
        end_datetime: SC DateTime of last record
        resolution: Data resolution (max 15 chars)
    """
    magic: bytes
    version: int
    symbol: str
    asset_type: str
    exchange: str
    provider: str
    bar_count: int
    start_datetime: float
    end_datetime: float
    resolution: str
    
    HEADER_SIZE: ClassVar[int] = 512
    STRUCT_FORMAT: ClassVar[str] = '<4sI32s16s32s32sQdd16s352s'
    
    def to_bytes(self) -> bytes:
        """Serialize header to 512 bytes.
        
        Returns:
            512-byte serialized header
            
        Raises:
            ValueError: If any string field exceeds maximum length
        """
        if len(self.symbol) > 31:
            raise ValueError(f"Symbol '{self.symbol}' exceeds 31 characters")
        if len(self.asset_type) > 15:
            raise ValueError(f"Asset type '{self.asset_type}' exceeds 15 characters")
        if len(self.exchange) > 31:
            raise ValueError(f"Exchange '{self.exchange}' exceeds 31 characters")
        if len(self.provider) > 31:
            raise ValueError(f"Provider '{self.provider}' exceeds 31 characters")
        if len(self.resolution) > 15:
            raise ValueError(f"Resolution '{self.resolution}' exceeds 15 characters")
        
        return struct.pack(
            self.STRUCT_FORMAT,
            self.magic,
            self.version,
            self.symbol.encode('utf-8').ljust(32, b'\x00'),
            self.asset_type.encode('utf-8').ljust(16, b'\x00'),
            self.exchange.encode('utf-8').ljust(32, b'\x00'),
            self.provider.encode('utf-8').ljust(32, b'\x00'),
            self.bar_count,
            self.start_datetime,
            self.end_datetime,
            self.resolution.encode('utf-8').ljust(16, b'\x00'),
            b'\x00' * 352
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'FileHeader':
        """Deserialize header from 512 bytes.
        
        Args:
            data: 512-byte serialized header
            
        Returns:
            FileHeader instance
            
        Raises:
            ValueError: If data is not exactly 512 bytes
        """
        if len(data) != cls.HEADER_SIZE:
            raise ValueError(
                f"Header must be exactly {cls.HEADER_SIZE} bytes, got {len(data)}"
            )
        
        unpacked = struct.unpack(cls.STRUCT_FORMAT, data)
        
        return cls(
            magic=unpacked[0],
            version=unpacked[1],
            symbol=unpacked[2].rstrip(b'\x00').decode('utf-8'),
            asset_type=unpacked[3].rstrip(b'\x00').decode('utf-8'),
            exchange=unpacked[4].rstrip(b'\x00').decode('utf-8'),
            provider=unpacked[5].rstrip(b'\x00').decode('utf-8'),
            bar_count=unpacked[6],
            start_datetime=unpacked[7],
            end_datetime=unpacked[8],
            resolution=unpacked[9].rstrip(b'\x00').decode('utf-8'),
        )


@dataclass
class DailyBar:
    """Daily OHLCV bar (48 bytes).
    
    Attributes:
        datetime: SC DateTime
        open: Opening price
        high: High price
        low: Low price
        close: Closing price
        volume: Volume
        open_interest: Open interest (0 for non-futures)
        adjustment_factor: Split/dividend factor (1.0 for non-equities)
        num_trades: Trade count (0 if unavailable)
        bid_volume: Bid-side volume (0 if unavailable)
        ask_volume: Ask-side volume (0 if unavailable)
    """
    datetime: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_interest: int
    adjustment_factor: float
    num_trades: int
    bid_volume: float
    ask_volume: float
    
    STRUCT_FORMAT: ClassVar[str] = '<dffffffififf'
    STRUCT_SIZE: ClassVar[int] = struct.calcsize(STRUCT_FORMAT)
    MAGIC_BYTES: ClassVar[bytes] = b'QKD1'
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes."""
        return struct.pack(
            self.STRUCT_FORMAT,
            self.datetime,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            self.open_interest,
            self.adjustment_factor,
            self.num_trades,
            self.bid_volume,
            self.ask_volume
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'DailyBar':
        """Deserialize from bytes.
        
        Args:
            data: 48-byte serialized bar
            
        Returns:
            DailyBar instance
            
        Raises:
            ValueError: If data is not exactly 48 bytes
        """
        if len(data) != cls.STRUCT_SIZE:
            raise ValueError(
                f"Expected {cls.STRUCT_SIZE} bytes, got {len(data)}"
            )
        unpacked = struct.unpack(cls.STRUCT_FORMAT, data)
        return cls(*unpacked)


@dataclass
class MinuteBar:
    """Intraday minute bar (48 bytes).
    
    Attributes:
        datetime: SC DateTime
        open: Opening price
        high: High price
        low: Low price
        close: Closing price
        volume: Volume
        tick_count: Number of ticks aggregated
        num_trades: Trade count (0 if unavailable)
        bid_volume: Bid-side volume (0 if unavailable)
        ask_volume: Ask-side volume (0 if unavailable)
        _padding: Reserved (always 0)
    """
    datetime: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_count: int
    num_trades: int
    bid_volume: float
    ask_volume: float
    _padding: int = 0
    
    STRUCT_FORMAT: ClassVar[str] = '<dffffffiiffi'
    STRUCT_SIZE: ClassVar[int] = struct.calcsize(STRUCT_FORMAT)
    MAGIC_BYTES: ClassVar[bytes] = b'QKM1'
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes."""
        return struct.pack(
            self.STRUCT_FORMAT,
            self.datetime,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            self.tick_count,
            self.num_trades,
            self.bid_volume,
            self.ask_volume,
            self._padding
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'MinuteBar':
        """Deserialize from bytes.
        
        Args:
            data: 48-byte serialized bar
            
        Returns:
            MinuteBar instance
            
        Raises:
            ValueError: If data is not exactly 48 bytes
        """
        if len(data) != cls.STRUCT_SIZE:
            raise ValueError(
                f"Expected {cls.STRUCT_SIZE} bytes, got {len(data)}"
            )
        unpacked = struct.unpack(cls.STRUCT_FORMAT, data)
        return cls(*unpacked)


@dataclass
class TickBar:
    """Individual tick/trade record (40 bytes).
    
    Attributes:
        datetime: SC DateTime (microsecond precision)
        price: Last trade price
        volume: Trade size
        bid: Best bid at time of trade
        ask: Best ask at time of trade
        bid_size: Size at best bid
        ask_size: Size at best ask
        flags: Trade condition flags
        _padding: Reserved (always 0)
    """
    datetime: float
    price: float
    volume: float
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    flags: int
    _padding: int = 0
    
    STRUCT_FORMAT: ClassVar[str] = '<dffffffIi'
    STRUCT_SIZE: ClassVar[int] = struct.calcsize(STRUCT_FORMAT)
    MAGIC_BYTES: ClassVar[bytes] = b'QKT1'
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes."""
        return struct.pack(
            self.STRUCT_FORMAT,
            self.datetime,
            self.price,
            self.volume,
            self.bid,
            self.ask,
            self.bid_size,
            self.ask_size,
            self.flags,
            self._padding
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'TickBar':
        """Deserialize from bytes.
        
        Args:
            data: 40-byte serialized tick
            
        Returns:
            TickBar instance
            
        Raises:
            ValueError: If data is not exactly 40 bytes
        """
        if len(data) != cls.STRUCT_SIZE:
            raise ValueError(
                f"Expected {cls.STRUCT_SIZE} bytes, got {len(data)}"
            )
        unpacked = struct.unpack(cls.STRUCT_FORMAT, data)
        return cls(*unpacked)
