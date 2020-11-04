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
import os

from forte.data.data_utils import maybe_download


def transform(ifile, ofile):
    # From https://github.com/XuezheMax/NeuroNLP2/issues/9
    dir_name = os.path.dirname(ofile)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    print(f"Converting to new file {ofile}")

    with open(ifile, 'r') as reader, open(ofile, 'w') as writer:
        prev = 'O'

        idx = 0
        for line in reader:
            line = line.strip()
            if len(line) == 0:
                prev = 'O'
                writer.write('\n')
                idx = 0
                continue

            tokens = line.split()
            # print tokens
            label = tokens[-1]
            if label != 'O' and label != prev:
                if prev == 'O':
                    label = 'B-' + label[2:]
                elif label[2:] != prev[2:]:
                    label = 'B-' + label[2:]
                else:
                    label = label
            idx += 1
            writer.write(f"{idx} " + " ".join(tokens[:-1]) + " " + label)
            writer.write('\n')
            prev = tokens[-1]


train_url = 'https://raw.githubusercontent.com/glample/tagger' \
            '/master/dataset/eng.train'
dev_url = 'https://raw.githubusercontent.com/glample/tagger' \
          '/master/dataset/eng.testa'
test_url = 'https://raw.githubusercontent.com/glample/tagger' \
           '/master/dataset/eng.testb'

filenames = [
    'train.conll', 'dev.conll', 'test.conll'
]

out_path = 'ner_data/conll03_english'

maybe_download(
    urls=[train_url, dev_url, test_url],
    path=os.path.join(out_path, 'raw'),
)

transform(os.path.join(out_path, 'raw', 'eng.testa'),
          os.path.join(out_path, 'dev', 'dev.conll'))

transform(os.path.join(out_path, 'raw', 'eng.testb'),
          os.path.join(out_path, 'test', 'test.conll'))

transform(os.path.join(out_path, 'raw', 'eng.train'),
          os.path.join(out_path, 'train', 'train.conll'))
