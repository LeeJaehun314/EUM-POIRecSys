# coding: utf-8
from sqlalchemy import ARRAY, Boolean, CHAR, Column, DateTime, Enum, ForeignKey, Integer, Numeric, String, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class KakaoCategoryMapping(Base):
    __tablename__ = 'kakao_category_mapping'

    id = Column(Integer, primary_key=True, server_default=text("nextval('kakao_category_mapping_id_seq'::regclass)"))
    kakao_category = Column(String, nullable=False, unique=True)
    ieum_category = Column(Enum('FOOD', 'CAFE', 'ALCOHOL', 'MUSEUM', 'STAY', 'SHOPPING', 'OTHERS', name='kakao_category_mapping_ieum_category_enum'), nullable=False)


class Place(Base):
    __tablename__ = 'place'

    id = Column(Integer, primary_key=True, server_default=text("nextval('place_id_seq'::regclass)"))
    name = Column(String, nullable=False)
    url = Column(String)
    address = Column(String, index=True)
    road_address = Column(String)
    kakao_id = Column(String)
    phone = Column(String)
    primary_category = Column(String, index=True)
    latitude = Column(Numeric)
    longitude = Column(Numeric)


class Tag(Base):
    __tablename__ = 'tag'

    id = Column(Integer, primary_key=True, server_default=text("nextval('tag_id_seq'::regclass)"))
    type = Column(Enum('0', '1', '2', name='tag_type_enum'), nullable=False, server_default=text("'2'::tag_type_enum"))
    tag_name = Column(String, nullable=False, index=True)


class User(Base):
    __tablename__ = 'user'

    created_at = Column(DateTime, nullable=False, server_default=text("now()"))
    updated_at = Column(DateTime, nullable=False, server_default=text("now()"))
    deleted_at = Column(DateTime)
    id = Column(Integer, primary_key=True, server_default=text("nextval('user_id_seq'::regclass)"))
    uuid = Column(UUID, nullable=False, server_default=text("uuid_generate_v4()"))
    o_auth_platform = Column(Enum('Apple', 'Kakao', 'Naver', name='user_o_auth_platform_enum'), nullable=False)
    o_auth_id = Column(String, nullable=False)
    nickname = Column(String)
    birth_date = Column(DateTime)
    sex = Column(CHAR(1))
    mbti = Column(String(4))
    jti = Column(String)
    is_ad_confirmed = Column(Boolean, nullable=False, server_default=text("false"))
    fcm_token = Column(String)


class Collection(Base):
    __tablename__ = 'collection'

    created_at = Column(DateTime, nullable=False, server_default=text("now()"))
    updated_at = Column(DateTime, nullable=False, server_default=text("now()"))
    deleted_at = Column(DateTime)
    id = Column(Integer, primary_key=True, server_default=text("nextval('collection_id_seq'::regclass)"))
    link = Column(String, nullable=False)
    content = Column(String)
    collection_type = Column(Enum('0', '1', name='collection_collection_type_enum'), nullable=False, server_default=text("'0'::collection_collection_type_enum"))
    user_id = Column(ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    is_viewed = Column(Boolean, nullable=False, server_default=text("false"))

    user = relationship('User')


class Folder(Base):
    __tablename__ = 'folder'

    id = Column(Integer, primary_key=True, server_default=text("nextval('folder_id_seq'::regclass)"))
    name = Column(String, nullable=False)
    user_id = Column(ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    type = Column(Enum('0', '1', '2', name='folder_type_enum'), nullable=False, server_default=text("'2'::folder_type_enum"))

    user = relationship('User')


class PlaceDetail(Base):
    __tablename__ = 'place_detail'

    id = Column(Integer, primary_key=True, server_default=text("nextval('place_detail_id_seq'::regclass)"))
    place_id = Column(ForeignKey('place.id', ondelete='CASCADE'), unique=True)
    week_days_opening_hours = Column(ARRAY(String()))
    free_parking_lot = Column(Boolean)
    paid_parking_lot = Column(Boolean)
    free_street_parking = Column(Boolean)
    google_maps_uri = Column(String)
    allows_dogs = Column(Boolean)
    good_for_groups = Column(Boolean)
    reservable = Column(Boolean)
    delivery = Column(Boolean)
    takeout = Column(Boolean)

    place = relationship('Place', uselist=False)


class PlaceImage(Base):
    __tablename__ = 'place_image'

    id = Column(Integer, primary_key=True, server_default=text("nextval('place_image_id_seq'::regclass)"))
    place_id = Column(ForeignKey('place.id', ondelete='SET NULL'), nullable=False)
    url = Column(String, nullable=False)
    author_name = Column(String)
    author_uri = Column(String)

    place = relationship('Place')


class PlaceTag(Base):
    __tablename__ = 'place_tag'

    id = Column(Integer, primary_key=True, server_default=text("nextval('place_tag_id_seq'::regclass)"))
    place_id = Column(ForeignKey('place.id', ondelete='CASCADE'), nullable=False)
    tag_id = Column(ForeignKey('tag.id', ondelete='CASCADE'), nullable=False)

    place = relationship('Place')
    tag = relationship('Tag')


class Preference(Base):
    __tablename__ = 'preference'

    id = Column(Integer, primary_key=True, server_default=text("nextval('preference_id_seq'::regclass)"))
    preferred_regions = Column(ARRAY(String()), nullable=False)
    preferred_companions = Column(ARRAY(String()), nullable=False)
    cheap_or_expensive = Column(Integer, nullable=False)
    planned_or_improvise = Column(Integer, nullable=False)
    tight_or_loose = Column(Integer, nullable=False)
    popular_or_local = Column(Integer, nullable=False)
    nature_or_city = Column(Integer, nullable=False)
    rest_or_activity = Column(Integer, nullable=False)
    user_id = Column(ForeignKey('user.id', ondelete='CASCADE'), unique=True)

    user = relationship('User', uselist=False)


class Trip(Base):
    __tablename__ = 'trip'

    id = Column(Integer, primary_key=True, server_default=text("nextval('trip_id_seq'::regclass)"))
    title = Column(String)
    companion_cnt = Column(Integer)
    companion_type = Column(ARRAY(String()), nullable=False)
    destination = Column(String)
    vehicle = Column(String)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    user_id = Column(ForeignKey('user.id'), nullable=False)

    user = relationship('User')


class AccommodationSchedule(Base):
    __tablename__ = 'accommodation_schedule'

    id = Column(Integer, primary_key=True, server_default=text("nextval('accommodation_schedule_id_seq'::regclass)"))
    name = Column(String, nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    trip_id = Column(ForeignKey('trip.id', ondelete='CASCADE'), nullable=False)

    trip = relationship('Trip')


class CollectionPlace(Base):
    __tablename__ = 'collection_place'

    id = Column(Integer, primary_key=True, server_default=text("nextval('collection_place_id_seq'::regclass)"))
    collection_id = Column(ForeignKey('collection.id', ondelete='CASCADE'), nullable=False)
    place_id = Column(ForeignKey('place.id'), nullable=False)
    is_saved = Column(Boolean, nullable=False, server_default=text("false"))
    place_keyword = Column(String, nullable=False)

    collection = relationship('Collection')
    place = relationship('Place')


class FolderPlace(Base):
    __tablename__ = 'folder_place'

    id = Column(Integer, primary_key=True, server_default=text("nextval('folder_place_id_seq'::regclass)"))
    folder_id = Column(ForeignKey('folder.id', ondelete='CASCADE'), nullable=False)
    place_id = Column(ForeignKey('place.id'), nullable=False)

    folder = relationship('Folder')
    place = relationship('Place')


class FolderTag(Base):
    __tablename__ = 'folder_tag'

    id = Column(Integer, primary_key=True, server_default=text("nextval('folder_tag_id_seq'::regclass)"))
    folder_id = Column(ForeignKey('folder.id', ondelete='CASCADE'), nullable=False)
    tag_id = Column(ForeignKey('tag.id', ondelete='CASCADE'), nullable=False)

    folder = relationship('Folder')
    tag = relationship('Tag')


class PlaceSchedule(Base):
    __tablename__ = 'place_schedule'

    id = Column(Integer, primary_key=True, server_default=text("nextval('place_schedule_id_seq'::regclass)"))
    visiting_date = Column(DateTime, nullable=False)
    place_id = Column(ForeignKey('place.id', ondelete='CASCADE'), nullable=False)
    trip_id = Column(ForeignKey('trip.id', ondelete='CASCADE'), nullable=False)

    place = relationship('Place')
    trip = relationship('Trip')


class TripStyle(Base):
    __tablename__ = 'trip_style'

    id = Column(Integer, primary_key=True, server_default=text("nextval('trip_style_id_seq'::regclass)"))
    accomodotion_location = Column(String, nullable=False)
    budget_style = Column(Integer, nullable=False)
    planning_style = Column(Integer, nullable=False)
    schedule_style = Column(Integer, nullable=False)
    destination_style1 = Column(Integer, nullable=False)
    destination_style2 = Column(Integer, nullable=False)
    destination_style3 = Column(Integer, nullable=False)
    trip_id = Column(ForeignKey('trip.id', ondelete='CASCADE'), unique=True)

    trip = relationship('Trip', uselist=False)
