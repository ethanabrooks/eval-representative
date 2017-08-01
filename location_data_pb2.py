# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: location_data.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='location_data.proto',
  package='work_project',
  syntax='proto2',
  serialized_pb=_b('\n\x13location_data.proto\x12\x0cwork_project\"\'\n\x07gLatLng\x12\r\n\x05latE7\x18\x01 \x01(\x05\x12\r\n\x05lngE7\x18\x02 \x01(\x05\"k\n\x0cLocationData\x12\x11\n\ttimestamp\x18\x01 \x01(\x05\x12%\n\x06latlng\x18\x02 \x01(\x0b\x32\x15.work_project.gLatLng\x12\x12\n\nis_optimal\x18\x03 \x01(\x08\x12\r\n\x05index\x18\x04 \x01(\x05')
)




_GLATLNG = _descriptor.Descriptor(
  name='gLatLng',
  full_name='work_project.gLatLng',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='latE7', full_name='work_project.gLatLng.latE7', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='lngE7', full_name='work_project.gLatLng.lngE7', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=37,
  serialized_end=76,
)


_LOCATIONDATA = _descriptor.Descriptor(
  name='LocationData',
  full_name='work_project.LocationData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='work_project.LocationData.timestamp', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='latlng', full_name='work_project.LocationData.latlng', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='is_optimal', full_name='work_project.LocationData.is_optimal', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='index', full_name='work_project.LocationData.index', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=78,
  serialized_end=185,
)

_LOCATIONDATA.fields_by_name['latlng'].message_type = _GLATLNG
DESCRIPTOR.message_types_by_name['gLatLng'] = _GLATLNG
DESCRIPTOR.message_types_by_name['LocationData'] = _LOCATIONDATA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

gLatLng = _reflection.GeneratedProtocolMessageType('gLatLng', (_message.Message,), dict(
  DESCRIPTOR = _GLATLNG,
  __module__ = 'location_data_pb2'
  # @@protoc_insertion_point(class_scope:work_project.gLatLng)
  ))
_sym_db.RegisterMessage(gLatLng)

LocationData = _reflection.GeneratedProtocolMessageType('LocationData', (_message.Message,), dict(
  DESCRIPTOR = _LOCATIONDATA,
  __module__ = 'location_data_pb2'
  # @@protoc_insertion_point(class_scope:work_project.LocationData)
  ))
_sym_db.RegisterMessage(LocationData)


# @@protoc_insertion_point(module_scope)
