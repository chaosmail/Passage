import pandoc
import os

pandoc.core.PANDOC_PATH = '/usr/bin/pandoc'

# Create New Pandoc Document
doc = pandoc.Document()

# Read the markdown README
doc.markdown = fs.read('README.md')

# Write a rST README for long_description
open('README.rst', 'w').write(doc.rst)

# Run the register command
os.system("python setup.py register sdist upload")

# Remove the rST README
os.unlink('README.rst')