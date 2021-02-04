import scipy.io.wavfile as wavfile
import os.path
import warnings
import numpy as np
import sofa
import json
import zipfile
import copy
import io
import sys

from pyfar import Signal
from pyfar import Coordinates
from pyfar.spatial.spatial import SphericalVoronoi
from pyfar.utils import str_to_type
import pyfar.dsp.classes as fo


def read_wav(filename):
    """
    Import a WAV file as signal object.

    This method is based on scipy.io.wavfile.read().

    Parameters
    ----------
    filename : string or open file handle
        Input wav file.

    Returns
    -------
    signal : signal instance
        An audio signal object from the pyfar Signal class
        containing the audio data from the WAV file.

    Notes
    -----
    * This function is based on scipy.io.wavfile.write().
    * This function cannot read wav files with 24-bit data.
    """
    sampling_rate, data = wavfile.read(filename)
    signal = Signal(data.T, sampling_rate, domain='time')
    return signal


def write_wav(signal, filename, overwrite=True):
    """
    Write a signal as a WAV file.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the pyfar Signal class.

    filename : string or open file handle
        Output wav file.

    overwrite : bool
        Select wether to overwrite the WAV file, if it already exists.
        The default is True.

    Notes
    -----
    * This function is based on scipy.io.wavfile.write().
    * Writes a simple uncompressed WAV file.
    * Signals of shape larger than 1D are flattened.
    * The bits-per-sample and PCM/float will be determined by the data-type.

    Common data types: [1]_

    =====================  ===========  ===========  =============
         WAV format            Min          Max       NumPy dtype
    =====================  ===========  ===========  =============
    32-bit floating-point  -1.0         +1.0         float32
    32-bit PCM             -2147483648  +2147483647  int32
    16-bit PCM             -32768       +32767       int16
    8-bit PCM              0            255          uint8
    =====================  ===========  ===========  =============

    Note that 8-bit PCM is unsigned.

    References
    ----------
    .. [1] IBM Corporation and Microsoft Corporation, "Multimedia Programming
       Interface and Data Specifications 1.0", section "Data Format of the
       Samples", August 1991
       http://www.tactilemedia.com/info/MCI_Control_Info.html

    """
    sampling_rate = signal.sampling_rate
    data = signal.time

    # Reshape to 2D
    data = data.reshape(-1, data.shape[-1])
    warnings.warn("Signal flattened to {data.shape[0]} channels.")

    # Check for .wav file extension
    if filename.split('.')[-1] != 'wav':
        warnings.warn("Extending filename by .wav.")
        filename += '.wav'

    # Check if file exists and for overwrite
    if overwrite is False and os.path.isfile(filename):
        raise FileExistsError(
            "File already exists,"
            "use overwrite option to disable error.")
    else:
        wavfile.write(filename, sampling_rate, data.T)


def read_sofa(filename):
    """
    Import a SOFA file as signal object.

    Parameters
    ----------
    filename : string or open file handle
        Input wav file.

    Returns
    -------
    signal : signal instance
        An audio signal object from the pyfar Signal class
        containing the IR data from the SOFA file with cshape being
        equal to (number of measurements, number of receivers).
    source_coordinates: coordinates instance
        An object from the pyfar Coordinates class containing
        the source coordinates from the SOFA file
        with matching domain, convention and unit.
    receiver_coordinates: coordinates instance
        An object from the pyfar Coordinates class containing
        the receiver coordinates from the SOFA file
        with matching domain, convention and unit.

    Notes
    -----
    * This function is based on the python-sofa package.
    * Only SOFA files of DataType 'FIR' are supported.

    References
    ----------
    .. [1] www.sofaconventions.org
    .. [2] “AES69-2015: AES Standard for File Exchange-Spatial Acoustic Data
       File Format.”, 2015.

    """
    sofafile = sofa.Database.open(filename)
    # Check for DataType
    if sofafile.Data.Type == 'FIR':
        domain = 'time'
        data = np.asarray(sofafile.Data.IR)
        sampling_rate = sofafile.Data.SamplingRate.get_values()
        # Check for units
        if sofafile.Data.SamplingRate.Units != 'hertz':
            raise ValueError(
                "SamplingRate:Units"
                "{sofafile.Data.SamplingRate.Units} is not supported.")
    else:
        raise ValueError("DataType {sofafile.Data.Type} is not supported.")
    signal = Signal(data, sampling_rate, domain=domain)

    # Source
    s_values = sofafile.Source.Position.get_values()
    s_domain, s_convention, s_unit = _sofa_pos(sofafile.Source.Position.Type)
    source_coordinates = Coordinates(
        s_values[:, 0],
        s_values[:, 1],
        s_values[:, 2],
        domain=s_domain,
        convention=s_convention,
        unit=s_unit)
    # Receiver
    r_values = sofafile.Receiver.Position.get_values()
    r_domain, r_convention, r_unit = _sofa_pos(sofafile.Receiver.Position.Type)
    receiver_coordinates = Coordinates(
        r_values[:, 0],
        r_values[:, 1],
        r_values[:, 2],
        domain=r_domain,
        convention=r_convention,
        unit=r_unit)

    return signal, source_coordinates, receiver_coordinates


def _sofa_pos(pos_type):
    if pos_type == 'spherical':
        domain = 'sph'
        convention = 'top_elev'
        unit = 'deg'
    elif pos_type == 'cartesian':
        domain = 'cart'
        convention = 'right'
        unit = 'met'
    else:
        raise ValueError("Position:Type {pos_type} is not supported.")
    return domain, convention, unit


def read(filename):
    """
    Read any compatible pyfar format from disk.

    Parameters
    ----------
    filename : string or open file handle.
        Input file must be haiopy compatible.

    Returns
    -------
    loaded_dict: dictionary containing haiopy types.
    """
    collection = {}
    with open(filename, 'rb') as f:
        zip_buffer = io.BytesIO()
        zip_buffer.write(f.read())
        with zipfile.ZipFile(zip_buffer) as zip_file:
            zip_paths = zip_file.namelist()
            obj_names = set([path.split('/')[0] for path in zip_paths])
            for name in obj_names:
                json_str = zip_file.read(name + '/json').decode('UTF-8')
                obj_type, obj_dict = json.loads(json_str)
                obj_dict = _decode(obj_dict, zip_file)
                ObjType = str_to_type(obj_type)
                collection[name] = ObjType._decode(obj_dict)
                

    return collection


def write(filename, compress=False, **objs):
    """
    Write any compatible pyfar format to disk.

    Parameters
    ----------
    filename : string or open file handle.
        Input file must be pyfar compatible.
    compress : bool
        Default is false (uncompressed).
        If false zipfile.ZIP_STORED mode is used,
        if True, zipfile.ZIP_DEFLATED mode is used.
    **objs: named compatible pyfar objects
        - Coordinates
        - Orientations
    """
    compression = zipfile.ZIP_STORED if compress else zipfile.ZIP_DEFLATED
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", compression) as zip_file:
        for name, obj in objs.items():
            obj_dict = _encode(copy.deepcopy(obj.__dict__), name, zip_file)
            type_obj_pair = [type(obj).__name__, obj_dict]
            zip_file.writestr(f'{name}/json', json.dumps(type_obj_pair))

    with open(filename, 'wb') as f:
        f.write(zip_buffer.getvalue())


def _decode(obj, zipfile):
    if isinstance(obj, dict):
        for key in obj.keys():
            _inner_decode(obj, key, zipfile)
    elif isinstance(obj, list):
        for i in range(0, len(obj)):
            _inner_decode(obj, i, zipfile)

    return obj


def _inner_decode(obj, key, zipfile):
    if not _is_type_hint(obj[key]):
        _decode(obj[key], zipfile)
    elif _is_mylib_type(obj[key][0]):
        MyType = str_to_type(obj[key][0])
        obj[key] = MyType._decode(obj[key][1])
    elif obj[key][0] == 'dtype':
        obj[key] = getattr(np, obj[key][1])
    elif obj[key][0] == 'ndarray':
        obj[key] = _decode_ndarray(obj[key][1], zipfile)


def _decode_ndarray(obj, zipfile):
    memfile = io.BytesIO()
    nd_bytes = zipfile.read(obj)
    memfile.write(nd_bytes)
    memfile.seek(0)
    return np.load(memfile, allow_pickle=False)


def _decode_sphericalvoronoi(obj_dict):    
    sampling = Coordinates(
        obj_dict['sampling'][:, 0],
        obj_dict['sampling'][:, 1],
        obj_dict['sampling'][:, 2],
        domain='cart')
    obj = SphericalVoronoi(
        sampling,
        center=obj_dict['center'])
    return obj


def _decode_filterFIR(PyfarType, obj_dict):
    obj = fo.FilterIIR(
        coefficients=obj_dict['_coefficients'][0, :, :],
        sampling_rate=obj_dict['_sampling_rate'])
    return obj


def _encode(obj, zip_path, zipfile):
    """
    Chooses the right encoding depending on the object type.

    Parameters
    ----------
    obj: Compatible Pyfar type.

    Returns
    ----------
    obj_dict_encoded: dict.
        Json compatible dictionary.
    obj_dict_ndarray: dict
        Numpy arrays are not JSON serializable thus encoded differently.
    """
    if isinstance(obj, dict):
        for key in obj.keys():
            _inner_encode(obj, key, f'{zip_path}/{key}', zipfile)
    elif isinstance(obj, list):
        for i in range(0, len(obj)):
            _inner_encode(obj, i, f'{zip_path}/{i}', zipfile)

    return obj


def _inner_encode(obj, key, zip_path, zipfile):
    if _is_dtype(obj[key]):
        obj[key] = ['dtype', obj[key].__name__]
    elif isinstance(obj[key], np.ndarray):
        zipfile.writestr(zip_path, _encode_ndarray(obj[key]))
        obj[key] = ['ndarray', zip_path]
    elif _is_mylib_type(obj[key]):
        obj[key] = [type(obj[key]).__name__, obj[key].__dict__]
        _encode(obj[key][1], zip_path, zipfile)
    else:
        _encode(obj[key], zip_path, zipfile)


def _is_mylib_type(obj):
    type_str = obj if isinstance(obj, str) else type(obj).__name__
    return type_str in [
        'NestedDataStruct', 'MyOtherClass']


def _is_dtype(obj):
    return isinstance(obj, type) and obj.__module__ == 'numpy'


def _is_type_hint(obj):
    return isinstance(obj, list) and len(obj) == 2


def _encode_sphericalvoronoi(obj):
    """
    The encoding of objects that are composed of primitive and numpy types
    utilizes `obj.__dict__()` and numpy encoding methods.

    Parameters
    ----------
    obj: Compatible Pyfar type.

    Returns
    ----------
    obj_dict_encoded: dict.
        Json compatible dictionary.
    obj_dict_ndarray: dict
        Numpy arrays are not JSON serializable thus encoded differently.
    """
    obj_dict_encoded = {}
    obj_dict_ndarray = {}
    obj_dict_encoded['type'] = type(obj).__name__
    obj_dict_ndarray['sampling'] = _encode_ndarray(obj.points)
    obj_dict_ndarray['center'] = _encode_ndarray(obj.center)
    return obj_dict_encoded, obj_dict_ndarray


def _encode_filter(obj):
    warnings.warn(f'`io.write` writing object of type {type(obj)}: ' 
        'It is not possible to save `filter_func` to disk.')
    obj_dict_encoded = {}
    obj_dict_ndarray = {}
    obj_dict_encoded['type'] = type(obj).__name__
    obj_dict_ndarray['_coefficients'] = _encode_ndarray(
        obj.__dict__['_coefficients'])
    obj_dict_encoded['_sampling_rate'] = obj.__dict__['_sampling_rate']
    obj_dict_encoded['_comment'] = obj.__dict__['_comment']
    obj_dict_ndarray['_state'] = _encode_ndarray(obj.__dict__['_state'])
    return obj_dict_encoded, obj_dict_ndarray
    

def _encode_ndarray(ndarray):
    """
    The encoding of objects that are composed of primitive and numpy types
    utilizes `obj.__dict__()` and numpy encoding methods.

    Parameters
    ----------
    ndarray: numpy.array.

    Returns
    -------
    bytes.
        They bytes that where written by `numpy.save` into a memfile.

    Notes
    -----
    * Do not allow pickling. It is not safe!
    """
    memfile = io.BytesIO()
    np.save(memfile, ndarray, allow_pickle=False)
    memfile.seek(0)
    return memfile.read()


def _unpack_zip_paths(zip_paths):
    """
    This is a helper function for read() to unpack zip-paths,
    e.g. 'coords/ndarrays/_points' and orientations/ndarrays/_quat
    becomes {'coords': [_points], 'orientations': '_quat'}

    Parameters
    ----------
    zip_paths: list of strings.
        Retreived zipfile.ZipFile.namelist()

    Returns
    ----------
    obj_paths: dict.
        Dictionary that contains unpacked zip paths.
    """
    obj_paths = {}
    for path in zip_paths:
        paths = path.split('/')
        obj_paths.setdefault(paths[0], [])
        if paths[1] == 'ndarrays':
            obj_paths[paths[0]].append(paths[2])
    return obj_paths
