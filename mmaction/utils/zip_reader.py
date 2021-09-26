import os
import zipfile

from mmcv import BaseStorageBackend, FileClient


class ZipReader(object):
    zip_bank = dict()

    def __init__(self):
        super(ZipReader, self).__init__()

    @staticmethod
    def is_zip_path(img_or_path):
        return '.zip@' in img_or_path

    @staticmethod
    def get_zipfile(path):
        zip_bank = ZipReader.zip_bank
        if path in zip_bank:
            return zip_bank[path]
        else:
            print('creating new zip_bank: {}'.format(path))
            zfile = zipfile.ZipFile(path, 'r')
            zip_bank[path] = zfile
            return zip_bank[path]

    @staticmethod
    def split_zip_style_path(path):
        pos_at = path.index('@')
        if pos_at == -1:
            print("character '@' is not found from the given path '%s'" %
                  (path))
            assert 0
        zip_path = path[0:pos_at]
        folder_path = path[pos_at + 1:]
        folder_path = str.strip(folder_path, '/')
        return zip_path, folder_path

    @staticmethod
    def list_folder(path):
        zip_path, folder_path = ZipReader.split_zip_style_path(path)

        zfile = ZipReader.get_zipfile(zip_path)
        folder_list = []
        for file_foler_name in zfile.namelist():
            file_foler_name = str.strip(file_foler_name, '/')
            if file_foler_name.startswith(folder_path) and \
               len(os.path.splitext(file_foler_name)[-1]) == 0 and \
               file_foler_name != folder_path:
                if len(folder_path) == 0:
                    folder_list.append(file_foler_name)
                else:
                    folder_list.append(file_foler_name[len(folder_path) + 1:])

        return folder_list

    @staticmethod
    def list_files(path, extension=['.*']):
        zip_path, folder_path = ZipReader.split_zip_style_path(path)

        zfile = ZipReader.get_zipfile(zip_path)
        file_lists = []
        for file_foler_name in zfile.namelist():
            file_foler_name = str.strip(file_foler_name, '/')
            if file_foler_name.startswith(folder_path) and str.lower(
                    os.path.splitext(file_foler_name)[-1]) in extension:
                if len(folder_path) == 0:
                    file_lists.append(file_foler_name)
                else:
                    file_lists.append(file_foler_name[len(folder_path) + 1:])

        return file_lists

    @staticmethod
    def read(path):
        zip_path, path_img = ZipReader.split_zip_style_path(path)
        zfile = ZipReader.get_zipfile(zip_path)
        data = zfile.read(path_img)
        return data


@FileClient.register_backend('zip')
class ZipBackend(BaseStorageBackend):
    """Zip storage backend (for internal use).

    Args:
        path_mapping (dict|None): path mapping dict from local path to Petrel
            path. When `path_mapping={'src': 'dst'}`, `src` in `filepath` will
            be replaced by `dst`. Default: None.
        enable_mc (bool): whether to enable memcached support. Default: True.
    """

    def __init__(self, path_mapping=None):
        assert isinstance(path_mapping, dict) or path_mapping is None
        self.path_mapping = path_mapping
        self.zip_bank = dict()

    def get_zipfile(self, path):
        if path in self.zip_bank:
            return self.zip_bank[path]
        else:
            zfile = zipfile.ZipFile(path, 'r')
            self.zip_bank[path] = zfile
            return zfile

    @staticmethod
    def split_zip_style_path(path):
        pos_at = path.index('@')
        if pos_at == -1:
            print("character '@' is not found from the given path '%s'" %
                  (path))
            assert 0
        zip_path = path[0:pos_at]
        folder_path = path[pos_at + 1:].strip('/')
        return zip_path, folder_path

    def get(self, filepath):
        filepath = str(filepath)
        if self.path_mapping is not None:
            for k, v in self.path_mapping.items():
                filepath = filepath.replace(k, v)
        zip_path, rest_path = self.split_zip_style_path(filepath)
        zfile = self.get_zipfile(zip_path)
        value = zfile.read(rest_path)
        value_buf = memoryview(value)
        return value_buf

    def get_text(self, filepath):
        raise NotImplementedError
