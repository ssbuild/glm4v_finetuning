# @Time    : 2023/1/22 16:22
# @Author  : tk
# @FileName: data_utils.py
import sys
import os

from torchvision import transforms

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import glob
from functools import cache
import copy
import json
import typing
import numpy as np
import torch
from deep_training.data_helper import DataHelper, ModelArguments, TrainingArguments, DataArguments, TrainingArgumentsHF, \
    TrainingArgumentsCL, TrainingArgumentsAC
from fastdatasets.record import load_dataset as Loader, RECORD, WriterObject, gfile
from tqdm import tqdm
from transformers import HfArgumentParser
from data_processer import DataStrategy, TokenIdsMaker
from deep_training.zoo.model_zoo.glm4v.llm_model import ChatGLM4Tokenizer,PetlArguments,ChatGLMConfig,build_masks_and_position_ids_glm
from config import *
from PIL import Image
from io import BytesIO


data_conf = {
   'strategy': DataStrategy.truncation, # 数据策略选项
    DataStrategy.truncation: {
    },
}


def preprocess(text):
  #text = text.replace("\n", "\\n").replace("\t", "\\t")
  return text

def postprocess(text):
  # return text.replace("\\n", "\n").replace("\\t", "\t")
  return text



class NN_DataHelper(DataHelper):
    index = 1
    def on_data_ready(self):
        self.index = -1


    def load_tokenizer_and_config(self, *args, **kwargs):
        ret = super().load_tokenizer_and_config(*args, **kwargs)
        tokenizer = self.tokenizer
        config = self.config
        self.image_size = tokenizer.image_size or 1120
        self.tokens_ids_maker = TokenIdsMaker(tokenizer=tokenizer, config=config)
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        return ret
        
    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        self.index += 1


        tokenizer: ChatGLM4Tokenizer
        config: ChatGLMConfig
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer
        config = self.config

        examples = data

        strategy = data_conf['strategy']
        if strategy == DataStrategy.truncation:
            ds = self.tokens_ids_maker.trunction(tokenizer, config, examples=examples, max_seq_length=max_seq_length,
                                                 **data_conf[strategy])
        # elif strategy == DataStrategy.siding:
        #     ds = self.tokens_ids_maker.slidding(tokenizer, config, examples=examples, max_seq_length=max_seq_length,
        #                                         **data_conf[strategy])
        else:
            raise ValueError('Invalid strategy', strategy)

        if not ds:
            return None

        if self.index < 3:
            print(ds[0])
        return ds

    def _get_messages(self, lines):
        D = []
        for line_id, line in enumerate(lines):
            jd = json.loads(line)
            if not jd:
                continue
            conversations = jd['conversations']
            if line_id < 10:
                print(conversations)

            cid = 0
            sub = []
            image_num = 0
            while cid < len(conversations):
                m = conversations[cid]
                cid += 1
                role = m["from"]
                q = preprocess(m["value"])
                img = m.get("img", None)
                if img is not None and img != "":
                    image_num += 1

                if role == "system":
                    assert len(sub) == 0
                    sub.append((role, q, m.pop('tools', None), img))
                    continue
                assert role in ['user', 'observation']
                m = conversations[cid]
                cid += 1
                assert m["from"] == "assistant"
                a = preprocess(m["value"])
                assert len(a), ValueError('answer cannot empty')
                sub.append((role, q, a, img))
            D.append(sub)
            assert image_num == 1, 'must has one image'
        return D

    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        D = []
        files = sum([glob.glob(file) for file in files], [])
        for file in files:
            with open(file, mode='r', encoding='utf-8', newline='\n') as f:
                lines = f.readlines()
            D.extend(self._get_messages(lines))
        return D

    def collate_fn(self, batch):
        batch = copy.copy(batch)
        o = {}
        for i, b in enumerate(batch):
            image_path = b.pop("img")
            image_path = image_path[0]
            if isinstance(image_path, bytes):
                image_path = str(image_path, encoding='utf-8')
            if image_path:
                image = Image.open(image_path)
                image = self.transform(image.convert('RGB'))
                b["images"] = image
            if i == 0:
                for k in b:
                    o[k] = [torch.tensor(b[k])]
            else:
                for k in b:
                    o[k].append(torch.tensor(b[k]))
        for k in o:
            o[k] = torch.stack(o[k])

        seqlens = o.pop('seqlen')
        max_len = torch.max(seqlens).tolist()
        input_ids = o['input_ids'][:, :max_len]

        attention_mask, position_ids = build_masks_and_position_ids_glm(input_ids, seqlens)
        o['input_ids'] = input_ids.long()
        o['attention_mask'] = attention_mask.long()
        o['position_ids'] = position_ids.long()
        o['labels'] = o['labels'][:, :max_len].long()
        return o

    def make_dataset_all(self):
        data_args = self.data_args
        # schema for arrow parquet
        schema = {
            "input_ids": "int32_list",
            "labels": "int32_list",
            "seqlen": "int32_list",
            "img": "binary_list",
        }

        # 缓存数据集
        if data_args.do_train:
            self.make_dataset_with_args(data_args.train_file, mixed_data=False, shuffle=True,
                                        mode='train', schema=schema)
        if data_args.do_eval:
            self.make_dataset_with_args(data_args.eval_file, mode='eval', schema=schema)
        if data_args.do_test:
            self.make_dataset_with_args(data_args.test_file, mode='test', schema=schema)

        # 记录缓存文件
        with open(os.path.join(data_args.output_dir, 'intermediate_file_index.json'), mode='w',
                  encoding='utf-8') as f:
            f.write(json.dumps({
                "train_files": self.train_files,
                "eval_files": self.eval_files,
                "test_files": self.test_files,
            }, ensure_ascii=False))

    @cache
    def load_dataset_files(self):
        data_args = self.data_args

        if not data_args.convert_file:
            return {
                "train_files": self.train_files,
                "eval_files": self.eval_files,
                "test_files": self.test_files,
            }

        filename = os.path.join(data_args.output_dir, 'intermediate_file_index.json')
        assert os.path.exists(filename), 'make you dataset firstly'
        with open(filename, mode='r', encoding='utf-8') as f:
            return json.loads(f.read())

if __name__ == '__main__':
    if global_args[ "trainer_backend" ] == "hf":
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsHF, DataArguments, PetlArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args = parser.parse_dict(config_args,
                                                                            allow_extra_keys=True, )
    elif global_args[ "trainer_backend" ] == "pl":
        parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, PetlArguments))
        model_args, training_args, data_args, _ = parser.parse_dict(config_args)
    elif global_args[ "trainer_backend" ] == "cl":
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsCL, DataArguments, PetlArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args = parser.parse_dict(config_args, allow_extra_keys=True, )
    else:
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsAC, DataArguments, PetlArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args = parser.parse_dict(config_args,
                                                                                         allow_extra_keys=True, )

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, _,_ = dataHelper.load_tokenizer_and_config(tokenizer_class_name=ChatGLM4Tokenizer,config_class_name=ChatGLMConfig)



    # 缓存数据集
    print(f'to make dataset is overwrite_cache {data_args.overwrite_cache}')
    dataHelper.make_dataset_all()

    print('make dataset complete!')
    print('check data !')
    dataset = dataHelper.load_sequential_sampler(dataHelper.load_dataset_files()["train_files"],
                                                 with_load_memory=data_args.data_backend == 'record',
                                                 batch_size=1,
                                                 collate_fn=dataHelper.collate_fn)

    print('total', len(dataset))
    for i, d in enumerate(dataset):
        print(d)
        if i > 3:
            break

