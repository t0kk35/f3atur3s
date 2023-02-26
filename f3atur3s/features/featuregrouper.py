"""
Definition of the Grouper feature. It is a feature directly found in a source.
(c) 2023 tsm
"""
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from functools import total_ordering
from datetime import timedelta, datetime
from typing import Optional, Dict, Any, List

from ..common.typechecking import enforce_types
from ..common.feature import Feature, FeatureWithBaseFeature
from ..common.learningcategory import LearningCategory
from .featurefilter import FeatureFilter


@enforce_types
@total_ordering
@dataclass(frozen=True, eq=False)
class TimePeriod(ABC):
    key: int = field(repr=False)
    name: str = field(compare=False)
    pandas_window: str = field(repr=False, compare=False)
    numpy_window: str = field(repr=False, compare=False)
    datetime_window: str = field(repr=False, compare=False)

    def __eq__(self, other):
        if isinstance(other, TimePeriod):
            return self.key == other.key
        else:
            raise TypeError(
                f'Can not check equality of TimePeriod object and non-TimePeriod object Got a {type(other)}'
            )

    def __lt__(self, other):
        if isinstance(other, TimePeriod):
            return self.key < other.key
        else:
            raise TypeError(
                f'Can not do < of TimePeriod object and non-TimePeriod object Got a {type(other)}'
            )

    def time_delta(self, number_of_periods: int):
        kw = {self.datetime_window: number_of_periods}
        return timedelta(**kw)

    @abstractmethod
    def delta_between(self, dt1: datetime, dt2: datetime) -> int:
        pass

    @abstractmethod
    def start_period(self, d: datetime) -> datetime:
        pass


@enforce_types
@dataclass(frozen=True)
class TimePeriodDay(TimePeriod):
    def delta_between(self, dt1: datetime, dt2: datetime) -> int:
        return (dt2 - dt1).days

    def start_period(self, d: datetime) -> datetime:
        # Remove time part
        return datetime(year=d.year, month=d.month, day=d.day)


@enforce_types
@dataclass(frozen=True)
class TimePeriodWeek(TimePeriod):
    def delta_between(self, dt1: datetime, dt2: datetime) -> int:
        return (dt2 - dt1).days // 7

    def start_period(self, d: datetime) -> datetime:
        # Go back to previous monday
        r = TIME_PERIOD_DAY.start_period(d)
        return r - timedelta(days=r.weekday())


@enforce_types
@dataclass(frozen=True)
class TimePeriodMonth(TimePeriod):
    def delta_between(self, dt1: datetime, dt2: datetime) -> int:
        return (dt2.year - dt1.year) * 12 + dt2.month - dt1.month

    def start_period(self, d: datetime) -> datetime:
        # Remove time and go to first day of month
        return datetime(year=d.year, month=d.month, day=1)


TIME_PERIOD_DAY = TimePeriodDay(0, 'Day', 'd', 'D', 'd')
TIME_PERIOD_WEEK = TimePeriodWeek(1, 'Week', 'w', 'W', 'w')
TIME_PERIOD_MONTH = TimePeriodMonth(2, 'Month', 'm', 'M', 'm')

TIME_PERIODS = [
    TIME_PERIOD_DAY,
    TIME_PERIOD_WEEK,
    TIME_PERIOD_MONTH
]

All_TIME_PERIODS = {
    t.key: t for t in TIME_PERIODS
}


class TimePeriodHelper:
    @classmethod
    def get_time_period(cls, key: int) -> TimePeriod:
        try:
            return All_TIME_PERIODS[key]
        except KeyError:
            raise FeatureDefinitionException(f'Could not find TimePeriod with key <{key}>')


@enforce_types
@dataclass(frozen=True, order=True)
class Aggregator:
    key: int = field(repr=False)
    name: str = field(compare=False)
    panda_agg_func: str = field(repr=False, compare=False)


AGGREGATOR_SUM = Aggregator(0, 'Sum', 'sum')
AGGREGATOR_COUNT = Aggregator(1, 'Count', 'count')
AGGREGATOR_MIN = Aggregator(2, 'Minimum', 'min')
AGGREGATOR_MAX = Aggregator(3, 'Maximum', 'max')
AGGREGATOR_AVG = Aggregator(4, 'Average', 'mean')
AGGREGATOR_STDDEV = Aggregator(5, 'Standard Deviation', 'std')

AGGREGATORS = [
    AGGREGATOR_SUM,
    AGGREGATOR_COUNT,
    AGGREGATOR_MIN,
    AGGREGATOR_MAX,
    AGGREGATOR_AVG,
    AGGREGATOR_STDDEV
]

ALL_AGGREGATORS = {
    a.key: a for a in AGGREGATORS
}


class AggregatorHelper:
    @classmethod
    def get_aggregator(cls, key: int) -> Aggregator:
        try:
            return ALL_AGGREGATORS[key]
        except KeyError:
            raise FeatureDefinitionException(f'Could not find Aggregator with key <{key}>')


@enforce_types
@dataclass(unsafe_hash=True, order=True)
class FeatureGrouper(FeatureWithBaseFeature):
    group_feature: Feature
    filter_feature: Optional[FeatureFilter] = field(compare=False)
    time_period: TimePeriod
    time_window: int
    aggregator: Aggregator

    def __post_init__(self):
        # Make sure the type float based.
        self.val_float_type()
        self.val_base_feature_is_float()
        # Embedded features are the base_feature, the group feature, the filter (if set) + their embedded features.
        eb = [self.group_feature, self.base_feature]
        eb.extend(self.group_feature.embedded_features + self.base_feature.embedded_features)
        if self.filter_feature is not None:
            eb.append(self.filter_feature)
            eb.extend(self.filter_feature.embedded_features)
        self.embedded_features = list(set(eb))

    def __dict__(self) -> Dict[str, Any]:
        json = super().__dict__()
        # Just need the name of the group_feature
        json['group_feature'] = json['group_feature']['name']
        # Just need the name of the filter_feature
        if 'filter_feature' in json:
            ff = json['filter_feature']
            if ff is not None:
                json['filter_feature'] = ff['name']
        return json

    @property
    def inference_ready(self) -> bool:
        # This feature is inference ready if all its embedded features are ready for inference.
        return all([f.inference_ready for f in self.embedded_features])

    @property
    def learning_category(self) -> LearningCategory:
        return self.type.learning_category

    @classmethod
    def create_from_save(cls, fields: Dict[str, Any], embedded_features: List['Feature'], pkl: Any) -> 'FeatureGrouper':
        name, ty, fb = cls.extract_dict(fields, embedded_features)
        fg = [f for f in embedded_features if f.name == fields['group_feature']][0]
        ff = [f for f in embedded_features if f.name == fields['filter_feature']]
        ff = ff[0] if len(ff) > 0 else None
        tp = TimePeriodHelper.get_time_period(fields['time_period']['key'])
        tw = fields['time_window']
        agg = AggregatorHelper.get_aggregator(fields['aggregator']['key'])
        return FeatureGrouper(name, ty, fb, fg, ff, tp, tw, agg)
