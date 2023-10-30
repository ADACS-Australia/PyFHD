import pytest
import numpy as np
from os import environ as env
from pathlib import Path
from PyFHD.gridding.gridding_utils import conjugate_mirror
from PyFHD.pyfhd_tools.test_utils import get_data_items
from PyFHD.io.pyfhd_io import save, load

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "conjugate_mirror")

@pytest.fixture(scope="function", params=[1,2,3])
def number(request):
    return request.param

@pytest.fixture
def conjugate_file(data_dir, number):
    conjugate_file = Path(data_dir, f"test_{number}_{data_dir.name}.h5")

    if conjugate_file.exists():
        return conjugate_file
    
    input, conj_mirror_image = get_data_items(data_dir, 
                                     f'visibility_grid_input_{number}.npy',
                                     f'visibility_grid_output_{number}.npy')
    
    h5_save_dict = {
        "input": input,
        "conj_mirror_image": conj_mirror_image
    }

    dd.io.save(conjugate_file, h5_save_dict)

    return conjugate_file

def test_conjugate_mirror(conjugate_file: Path):
    h5 = load(conjugate_file)
    image = conjugate_mirror(h5["input"])
    assert np.array_equal(image, h5["conj_mirror_image"])