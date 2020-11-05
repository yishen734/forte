#  Copyright 2020 The Forte Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import zipfile
import logging

from forte.data.data_utils import maybe_download

print("Downloading glove data.")
maybe_download(
    urls=['http://nlp.stanford.edu/data/glove.6B.zip'],
    path='ner_data'
)
print("Done.")

print("Unzip data.")
with zipfile.ZipFile('ner_data/glove.6B.zip', 'r') as zip_ref:
    zip_ref.extractall('ner_data/glove.6B')
print("Done.")
