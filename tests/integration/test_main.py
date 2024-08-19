import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from subprocess import Popen, PIPE

def test_main_script_execution():
    # Modify the config file to run only 100 epochs for testing purposes
    with open('src/config.py', 'r') as file:
        config_lines = file.readlines()
    
    with open('src/config.py', 'w') as file:
        for line in config_lines:
            if "num_epochs" in line:
                file.write('    "num_epochs": 100,\n')
            else:
                file.write(line)

    try:
        # Run the main script
        process = Popen(['python', 'src/main.py'], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()

        assert process.returncode == 0, f"Main script exited with an error:\n{stderr.decode()}"
    finally:
        # Revert the config file back to original state after the test
        with open('src/config.py', 'w') as file:
            file.writelines(config_lines)

