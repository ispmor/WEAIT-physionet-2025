import h5py
import numpy as np
from pathlib import Path
import torch
from torch.utils import data


class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.

    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """

    def __init__(self, file_path, recursive, load_data, data_cache_size=3, transform=None, leads=np.zeros(1)):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform
        self.files = None
        self.leads = leads

        # Search for all h5 files
        p = Path(file_path)
        if p.is_dir():
            if recursive:
                self.files = sorted(p.glob('**/*.h5'))
            else:
                self.files = sorted(p.glob('*.h5'))
        elif p.is_file():
            self.files = [p]

        if len(self.files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in self.files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)

    def open_hdf5(self):
        self.file = h5py.File(self.files[0], "r")

    def __getitem__(self, index):
        if not hasattr(self, 'file'):
            self.open_hdf5()
        # get data
        x_raw = self.get_data("data", index)
        x_drift_removed = self.get_data("drift_removed", index)
        x_bw_removed = self.get_data("bw_removed", index)


        if self.transform:
            x_raw = self.transform(x_raw)
            x_drift_removed = self.transform(x_drift_removed)
            x_bw_removed = self.transform(x_bw_removed)
        else:
            x_raw = torch.from_numpy(np.array(x_raw))
            x_drift_removed = torch.from_numpy(np.array(x_drift_removed))
            x_bw_removed = torch.from_numpy(np.array(x_bw_removed))

        # get label
        y = self.get_data("label", index)
        y = torch.from_numpy(np.array(y))

        rr_features = self.get_data("rr_features", index)
        rr_features = torch.from_numpy(np.array(rr_features))

        wavelet_features = self.get_data("wavelet_features", index)
        wavelet_features = torch.from_numpy(np.array(wavelet_features))

        return (x_raw, x_drift_removed, x_bw_removed, y, rr_features, wavelet_features)


        #TODO: dodać zwracanie array wavelet i może r features?

    def __len__(self):
        return self.get_data_infos('data')[0]['shape'][0]

    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path) as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # if data is not loaded its cache index is -1
                    idx = -1
                    if load_data:
                        # add data to the data cache
                        idx = self._add_to_cache(ds.value, file_path)

                    # type is derived from the name of the dataset; we expect the dataset
                    # name to have a name such as 'data' or 'label' to identify its type
                    # we also store the shape of the data in case we need it
                    self.data_info.append(
                        {'file_path': file_path, 'type': dname, 'shape': ds.shape, 'cache_idx': idx})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        # with h5py.File(file_path) as h5_file:
        # for gname, group in h5_file.items():
        h5_file = self.file
        for gname, group in h5_file.items():  # group.items():
            for dname, ds in group.items():
                # add data to the data cache and retrieve
                # the cache index
                idx = self._add_to_cache(ds, file_path)

                # find the beginning index of the hdf5 file we are looking for
                file_idx = next(i for i, v in enumerate(self.data_info) if v['file_path'] == file_path)

                # the data info should have the same index since we loaded it in the same way
                self.data_info[file_idx + idx]['cache_idx'] = idx


        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [
                {'file_path': di['file_path'],
                 'type': di['type'],
                 'shape': di['shape'],
                 'cache_idx': -1} if di['file_path'] == removal_keys[0] else di
                for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[0]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)

        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[0]['cache_idx']
        if type in "label":
            return self.data_cache[fp][cache_idx][i]
        else:
            return self.data_cache[fp][cache_idx][i][self.leads]
