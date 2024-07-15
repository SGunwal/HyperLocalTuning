# utils/installs.py

import subprocess
import sys
import pkg_resources

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of packages to install
packages = [
    'pandas',
    'numpy',
    'gurobipy',
    'math',
    'tensorflow',
    'optuna',
    'pickle',  # pkl is a module, but it's referred to as pickle5 for installation
    'datetime',
    'matplotlib',
    'seaborn',
    'scikit-learn'
]

# Install each package
for package in packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

# Get installed versions of packages
installed_packages = {package.key: package.version for package in pkg_resources.working_set}

# Write installed packages and versions to versions.txt
with open('./utils/versions.txt', 'w') as f:
    for package, version in installed_packages.items():
        f.write(f"{package}=={version}\n")

print("Installation complete. Versions of installed packages have been written to versions.txt.")