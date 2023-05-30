from tqdm import tqdm
from ._base import BaseTask
import torch
import numpy as np
from typing import Any, Dict, List, Optional
from datasets import load_dataset

class PerplexityTask(BaseTask):
    def __init__(
        self,
        model,
        tokenizer,
        dataset='wikitext',
        n_ctx = 512,
        n_batch = 512,
        quiet = False,
        device: Optional[str] = None,
        **kwargs
    ):
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.quiet = quiet

        prompt_col_name = 'text'

        def preprocess_fn(samples):
            return {prompt_col_name: ' \n' if samples[prompt_col_name] == '' else samples[prompt_col_name]}
        
        def get_wikitext(blah, **kwargs):
            wikidata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
            def blah(sample):
                return { "text": ' \n' if sample['text'] == '' else sample['text']}
            wikidata.map(blah)
            return wikidata

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            data_name_or_path="wikitext",
            prompt_col_name="text",
            batched=False,
            device=device,
            preprocess_fn=preprocess_fn,
            load_fn = get_wikitext,
            #load_fn_kwargs = { "name": 'wikitext-2-raw-v1', "split": "test"},
            truncate_prompt=False,
            add_eos_token=False,
            add_special_tokens=False,
            limit_samples=False,
            **kwargs
        )

    @staticmethod
    def softmax(logits):
        e_x = np.exp(logits - np.max(logits))
        return e_x / e_x.sum(axis=0)

    def _predict(self, batch_data: Dict[str, Any], *args, **kwargs) -> List[float]:
        pass

    def run(self) -> Dict[str, float]:
        tokens = torch.LongTensor().to(self.device)  # initialize an empty tensor
        for item in self.dl:
            input_ids = torch.stack(item['input_ids'][0]).reshape(1, -1)
            tokens = torch.cat((tokens, input_ids.to(self.device)), dim=1)  # concatenate the new tensor

        # Convert the tensor to a list of strings
        tokens_list = [str(t.item()) for t in tokens[0]]

        # Join the list into a single string
        dump_tokens = ','.join(tokens_list)

        # Write the string to the file
        with open('dump_tokens.new', 'w') as f:
            f.write(dump_tokens)
            
        text_decode = self.tokenizer.decode(tokens[0])
        with open('decode_tokens.new', 'w') as f:
            f.write(text_decode)

        len_tokens = len(tokens[0])
        print("Length of data:", len_tokens)
        n_ctx = self.n_ctx
        n_batch = self.n_batch
        n_chunk = len_tokens // n_ctx

        nll = 0.0
        count = 0

        # Algorithm duplicated from llama.cpp's perplexity so that results can be compared to the many ppl figures published already
        # https://github.com/ggerganov/llama.cpp/blob/master/examples/perplexity/perplexity.cpp
        print(f'Calculating perplexity over {n_chunk} chunks, batch_size={n_batch}')

        progress = tqdm(range(n_chunk))
        progress.set_description(f"Perplexity: - ")
        for i in progress:
            start = i * n_ctx
            end = start + n_ctx

            num_batches = (n_ctx + n_batch - 1) // n_batch

            logits = []

            for j in range(num_batches):
                batch_start = start + j * n_batch
                batch_size  = min(end - batch_start, n_batch)

                token_org = tokens[0][batch_start].item()

                if j == 0:
                    tokens[0][batch_start] = self.tokenizer.bos_token_id

                with torch.no_grad():
                    outputs = self.model(tokens[:, batch_start:batch_start+batch_size])
                    batch_logits = outputs.logits.float()

                tokens[0][batch_start] = token_org

                logits.append(batch_logits.detach())

            for j in range(min(512, n_ctx // 2), n_ctx - 1):
                tok_logits = logits[0][0][j].cpu().numpy()
                prob = self.softmax(tok_logits)[tokens[0][start + j + 1]]

                nll += -np.log(prob)
                count += 1

            ppl = np.exp(nll / count)

            #if not self.quiet:
            #    progress.write(f'[{i+1}]{ppl:.4f}, ', end='')
            progress.set_description(f"Perplexity: {ppl:.4f}")

        print(f"Perplexity: {ppl:.4f}")
        #print(results)