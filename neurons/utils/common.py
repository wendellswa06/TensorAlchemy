import os
import sys


def is_validator() -> bool:
    main_module = sys.modules["__main__"]
    main_file = os.path.abspath(main_module.__file__)
    return "neurons/validator" in main_file
