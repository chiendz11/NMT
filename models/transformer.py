import torch
import torch.nn as nn
import torchtext.data as data
import copy, time, io
import numpy as np

from modules.prototypes import Encoder, Decoder, Config as DefaultConfig
from modules.loader import DefaultLoader, MultiLoader
from modules.config import MultiplePathConfig as Config
from modules.inference import strategies
from modules import constants as const
from modules.optim import optimizers, ScheduledOptim

import utils.save as saver
from utils.decode_old import create_masks, translate_sentence
from utils.loss import LabelSmoothingLoss
from utils.metric import bleu, bleu_batch_iter, bleu_single, bleu_batch

# ==============================================================================
# CLASS EARLY STOPPING
# ==============================================================================
class EarlyStopping:
    """
    Dừng training sớm nếu validation loss không giảm sau 'patience' epochs.
    """
    def __init__(self, patience=10, delta=0.0001, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, logging=None):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            if logging and self.verbose:
                logging.info(f'[EarlyStopping] Counter: {self.counter}/{self.patience} (Best Val Loss: {self.val_loss_min:.4f})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0

# ==============================================================================
# TRANSFORMER MODEL
# ==============================================================================
class Transformer(nn.Module):
    """
    Implementation of Transformer architecture based on the paper Attention is all you need.
    Updated for 8GB VRAM Optimization Strategy (Cache Clearing enabled).
    """
    def __init__(self, mode=None, model_dir=None, config=None):
        super().__init__()
        # 1. LOAD CONFIG & PARAMS
        self.config = DefaultConfig() if(config is None) else Config(config)
        opt = self.config
        self.device = opt.get('device', const.DEFAULT_DEVICE)

        # Param quan trọng để Loader lọc dữ liệu
        self.train_max_length = opt.get('train_max_length', const.DEFAULT_TRAIN_MAX_LENGTH)

        # 2. KHỞI TẠO LOADER
        if('train_data_location' in opt or 'train_data_location' in opt.get("data", {})):
            data_opt = opt if 'train_data_location' in opt else opt["data"]
            self.loader = DefaultLoader(
                data_opt['train_data_location'], 
                eval_path=data_opt.get('eval_data_location', None), 
                language_tuple=(data_opt["src_lang"], data_opt["trg_lang"]), 
                option=opt
            )
        elif('data' in opt):
            self.loader = MultiLoader(opt["data"]["train"], valid=opt["data"].get("valid", None), option=opt)
        
        self.SRC, self.TRG = self.loader.build_field(lower=opt.get("lowercase", const.DEFAULT_LOWERCASE))

        # 3. TẠO ITERATOR
        if(mode == "train"):
            self.train_iter, self.valid_iter = self.loader.create_iterator(
                self.fields, 
                model_path=model_dir
            )
        elif(mode == "eval"):
            self.train_iter, self.valid_iter = self.loader.create_iterator(self.fields, model_path=model_dir)
        elif(mode == "infer"):
            self.loader.build_vocab(self.fields, model_path=model_dir)
        else:
            raise ValueError("Unknown model's mode: {}".format(mode))

        # 4. KHỞI TẠO MODEL NETWORK
        src_vocab_size, trg_vocab_size = len(self.SRC.vocab), len(self.TRG.vocab)
        d_model, N, heads, dropout = opt['d_model'], opt['n_layers'], opt['heads'], opt['dropout']
        
        self.encoder = Encoder(src_vocab_size, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab_size, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab_size)

        # Configs cho Inference
        input_max_length = self.config.get("input_max_length", const.DEFAULT_INPUT_MAX_LENGTH)
        infer_max_length = self.config.get('max_length', const.DEFAULT_MAX_LENGTH)
        
        decode_strategy_class = strategies[opt.get('decode_strategy', const.DEFAULT_DECODE_STRATEGY)]
        decode_strategy_kwargs = opt.get('decode_strategy_kwargs', const.DEFAULT_STRATEGY_KWARGS)
        self.decode_strategy = decode_strategy_class(self, infer_max_length, self.device, **decode_strategy_kwargs)

        self.to(self.device)

    def load_checkpoint(self, model_dir, checkpoint=None, checkpoint_idx=0):
        if(checkpoint is not None):
            saver.load_model(self, checkpoint)
            self._checkpoint_idx = checkpoint_idx
        else:
            if model_dir is not None:
                checkpoint_idx = saver.check_model_in_path(model_dir)
                if(checkpoint_idx > 0):
                    print("Found model with index {:d} already saved.".format(checkpoint_idx))
                    saver.load_model_from_path(self, model_dir, checkpoint_idx=checkpoint_idx)
                else:
                    print("No checkpoint found, start from beginning.")
                    checkpoint_idx = -1
            else:
                print("No model_dir available, start from beginning.")
                checkpoint_idx = -1
            self._checkpoint_idx = checkpoint_idx
            
    def forward(self, src, trg, src_mask, trg_mask, output_attention=False):
        e_outputs = self.encoder(src, src_mask)
        d_output, attn = self.decoder(trg, e_outputs, src_mask, trg_mask, output_attention=True)
        output = self.out(d_output)
        if(output_attention):
            return output, attn
        else:
            return output

    # --- TRAIN STEP ---
    def train_step(self, optimizer, batch, criterion, accum_count=1, step=0):
        self.train()
        opt = self.config
        
        src = batch.src.transpose(0, 1).to(opt.get('device', const.DEFAULT_DEVICE))
        trg = batch.trg.transpose(0, 1).to(opt.get('device', const.DEFAULT_DEVICE))

        trg_input = trg[:, :-1]
        src_pad = self.SRC.vocab.stoi['<pad>']
        trg_pad = self.TRG.vocab.stoi['<pad>']
        ys = trg[:, 1:].contiguous().view(-1)

        src_mask, trg_mask = create_masks(src, trg_input, src_pad, trg_pad, opt.get('device', const.DEFAULT_DEVICE))
        preds = self(src, trg_input, src_mask, trg_mask)
        
        loss = criterion(preds.view(-1, preds.size(-1)), ys)
        
        # Normalize Loss theo accum_count
        loss = loss / accum_count 
        loss.backward()
        
        # Chỉ update trọng số khi đã tích đủ batch
        if (step + 1) % accum_count == 0:
            optimizer.step_and_update_lr()
            optimizer.zero_grad()
        
        return loss.item() * accum_count    

    def validate(self, valid_iter, criterion, maximum_length=None):
        self.eval()
        opt = self.config
        src_pad = self.SRC.vocab.stoi['<pad>']
        trg_pad = self.TRG.vocab.stoi['<pad>']
    
        with torch.no_grad():
            total_loss = []
            for batch in valid_iter:
                src = batch.src.transpose(0, 1).to(opt.get('device', const.DEFAULT_DEVICE))
                trg = batch.trg.transpose(0, 1).to(opt.get('device', const.DEFAULT_DEVICE))
                
                # Cắt dữ liệu validation theo max_length để đồng bộ & an toàn bộ nhớ
                if(maximum_length is not None):
                    src = src[:, :min(src.shape[1], maximum_length[0])]
                    trg = trg[:, :min(trg.shape[1], maximum_length[1])]
                
                trg_input = trg[:, :-1]
                ys = trg[:, 1:].contiguous().view(-1)

                src_mask, trg_mask = create_masks(src, trg_input, src_pad, trg_pad, opt.get('device', const.DEFAULT_DEVICE))
                preds = self(src, trg_input, src_mask, trg_mask)

                loss = criterion(preds.view(-1, preds.size(-1)), ys)
                total_loss.append(loss.item())
    
        return np.mean(total_loss)

    def translate_sentence(self, sentence, device=None, k=None, max_len=None, debug=False):
        self.eval()
        if(device is None): device = self.config.get('device', const.DEFAULT_DEVICE)
        if(k is None): k = self.config.get('k', const.DEFAULT_K)
        if(max_len is None): max_len = self.config.get('max_length', const.DEFAULT_MAX_LENGTH)
        translated_tokens = translate_sentence(sentence, self, self.SRC, self.TRG, device, k, max_len, debug=debug, output_list_of_tokens=True)
        return translated_tokens

    def translate_batch_sentence(self, sentences, src_lang=None, trg_lang=None, output_tokens=False, batch_size=None):
        if(batch_size is None): 
            batch_size = self.config.get("eval_batch_size", const.DEFAULT_EVAL_BATCH_SIZE)
        input_max_length = self.config.get("input_max_length", const.DEFAULT_INPUT_MAX_LENGTH)
        self.eval()

        translated = []
        for b_idx in range(0, len(sentences), batch_size):
            batch = sentences[b_idx: b_idx+batch_size]
            trans_batch = self.translate_batch(batch, trg_lang=trg_lang, output_tokens=output_tokens, input_max_length=input_max_length)
            translated.extend(trans_batch)
            for line in trans_batch:
                print(line)
        return translated

    def translate_batch(self, batch_sentences, src_lang=None, trg_lang=None, output_tokens=False, input_max_length=None):
        if(input_max_length is None):
            input_max_length = self.config.get("input_max_length", const.DEFAULT_INPUT_MAX_LENGTH)
        translated_batch = self.decode_strategy.translate_batch(batch_sentences, trg_lang=trg_lang, src_size_limit=input_max_length, output_tokens=True, debug=False)
        return self.loader.detokenize(translated_batch) if not output_tokens else translated_batch

    # ==========================================================================
    # RUN TRAIN: LOGIC CHÍNH
    # ==========================================================================
    def run_train(self, model_dir=None, config=None):
        opt = self.config
        from utils.logging import init_logger
        logging = init_logger(model_dir, opt.get('log_file_models'))

        trg_pad = self.TRG.vocab.stoi['<pad>']     
        logging.info("%s * src vocab size = %s"%(self.loader._language_tuple[0] ,len(self.SRC.vocab)))
        logging.info("%s * tgt vocab size = %s"%(self.loader._language_tuple[1] ,len(self.TRG.vocab)))
        logging.info("Building model...")
        model = self.to(opt.get('device', const.DEFAULT_DEVICE))

        checkpoint_idx = self._checkpoint_idx
        if(checkpoint_idx < 0):
            print("Zero checkpoint detected, reinitialize the model")
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            checkpoint_idx = 0

        best_model_score = saver.load_model_score(model_dir)
        
        # SETUP OPTIMIZER
        optim_algo = opt["optimizer"]
        lr = opt["lr"]
        d_model = opt["d_model"]
        n_warmup_steps = opt.get("n_warmup_steps", 4000)
        optimizer_params = opt.get("optimizer_params", dict({}))

        if optim_algo not in optimizers:
            raise ValueError("Unknown optimizer: {}".format(optim_algo))
        
        optimizer = ScheduledOptim(
                optimizer=optimizers.get(optim_algo)(model.parameters(), **optimizer_params),
                init_lr=lr, 
                d_model=d_model, 
                n_warmup_steps=n_warmup_steps
            )
        
        criterion = LabelSmoothingLoss(len(self.TRG.vocab), padding_idx=trg_pad, smoothing=opt['label_smoothing'])
        
        # Logging Info
        params_encode = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, self.encoder.parameters())])
        params_decode = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, self.decoder.parameters())])
        logging.info("* Number of parameters: %s"%(params_encode+params_decode))
        logging.info("Starting training on %s"%(opt.get('device', const.DEFAULT_DEVICE)))
        
        # REPORT CHIẾN THUẬT TRAIN
        accum_count = opt.get('accum_count', 1)
        batch_size = opt.get('batch_size', 1) 
        effective_batch_size = batch_size * accum_count
        
        logging.info("------------------------------------------------------")
        logging.info(" TRAINING STRATEGY REPORT")
        logging.info(f" * Physical Batch Size : {batch_size}")
        logging.info(f" * Accumulation Steps  : {accum_count}")
        logging.info(f" * EFFECTIVE BATCH SIZE: {effective_batch_size}")
        logging.info(f" * Train Max Length    : {self.train_max_length}")
        logging.info(f" * Warmup Steps        : {n_warmup_steps}")
        logging.info("------------------------------------------------------")
        
        # Cấu hình Early Stopping
        patience = opt.get('patience', 10) 
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        logging.info(f"Early Stopping enabled: patience={patience}")

        optimizer.zero_grad()

        for epoch in range(checkpoint_idx, opt['epochs']):
            total_loss = 0.0
            
            s = time.time()
            for i, batch in enumerate(self.train_iter): 
                # Chạy train step với accumulation
                loss = self.train_step(optimizer, batch, criterion, accum_count=accum_count, step=i)
                total_loss += loss
                
                if (i + 1) % opt['printevery'] == 0:
                    avg_loss = total_loss / opt['printevery']
                    et = time.time() - s
                    
                    if hasattr(optimizer, 'get_last_lr'):
                        curr_lr = optimizer.get_last_lr()
                    else:
                        curr_lr = optimizer._optimizer.param_groups[0]['lr']

                    logging.info('epoch: {:03d} - iter: {:05d} - train loss: {:.4f} - lr: {:.8f} - time: {:.4f}'.format(
                        epoch, i+1, avg_loss, curr_lr, et))
                    total_loss = 0
                    s = time.time()
            
            # Update cuối epoch nếu batch lẻ (những mẫu cuối cùng chưa được update)
            if (i + 1) % accum_count != 0:
                optimizer.step_and_update_lr()
                optimizer.zero_grad()
            
            # ==================================================================
            # OPTIMIZATION: CLEAR CACHE TRƯỚC KHI VALIDATE
            # ==================================================================
            # Giải phóng VRAM không dùng đến để tránh OOM khi vào Validate
            if 'cuda' in str(opt.get('device', 'cpu')):
                torch.cuda.empty_cache()
                # logging.info("Cleared VRAM cache before validation.")
            # ==================================================================

            # --- Validation Phase ---
            s = time.time()
            valid_loss = self.validate(
                self.valid_iter, 
                criterion, 
                maximum_length=(self.train_max_length, self.train_max_length)
            )
            
            # --- Saving Logic ---
            if (epoch+1) % opt['save_checkpoint_epochs'] == 0 and model_dir is not None:
                valid_src_lang, valid_trg_lang = self.loader.language_tuple
                bleuscore = bleu_batch_iter(self, self.valid_iter, src_lang=valid_src_lang, trg_lang=valid_trg_lang)

                saver.save_and_clear_model(model, model_dir, checkpoint_idx=epoch+1, maximum_saved_model=opt.get('maximum_saved_model_train', const.DEFAULT_NUM_KEEP_MODEL_TRAIN))
                best_model_score = saver.save_model_best_to_path(model, model_dir, best_model_score, bleuscore, maximum_saved_model=opt.get('maximum_saved_model_eval', const.DEFAULT_NUM_KEEP_MODEL_TRAIN))
                logging.info('epoch: {:03d} - valid loss: {:.4f} - bleu: {:.4f} - time: {:.4f}'.format(epoch, valid_loss, bleuscore, time.time() - s))
            else:
                logging.info('epoch: {:03d} - valid loss: {:.4f} - time: {:.4f}'.format(epoch, valid_loss, time.time() - s))

            # CHECK EARLY STOPPING
            early_stopping(valid_loss, logging)
            
            if early_stopping.early_stop:
                logging.info("Early stopping triggered! Training stopped.")
                saver.save_model_name(type(self).__name__, model_dir)
                break
    
    def run_infer(self, features_file, predictions_file, src_lang=None, trg_lang=None, config=None, batch_size=None):
        opt = self.config
        model = self.to(opt.get('device', const.DEFAULT_DEVICE))
        
        print("Reading features file from {}...".format(features_file))
        with io.open(features_file, "r", encoding="utf-8") as read_file:
            inputs = [l.strip() for l in read_file.readlines()]
        
        print("Performing inference ...")
        start = time.time()
        results = "\n".join( self.translate_batch_sentence(inputs, src_lang=src_lang, trg_lang=trg_lang, output_tokens=False, batch_size=batch_size))
        print("Inference done, cost {:.2f} secs.".format(time.time() - start))

        print("Writing results to {} ...".format(predictions_file))
        with io.open(predictions_file, "w", encoding="utf-8") as write_file:
            write_file.write(results)

        print("All done!")

    
    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def to_logits(self, inputs): 
        return self.out(inputs)

    def prepare_serve(self, serve_path, model_dir=None, check_trace=True, **kwargs):
        self.eval()
        saver.save_model_name(type(self).__name__, model_dir)
        fake_batch, fake_srclen, fake_trglen, fake_range = 3, 7, 4, 1000
        sample_src, sample_trg = torch.randint(fake_range, (fake_batch, fake_srclen), dtype=torch.long), torch.randint(fake_range, (fake_batch, fake_trglen), dtype=torch.long)
        sample_src_mask, sample_trg_mask = torch.rand(fake_batch, 1, fake_srclen) > 0.5, torch.rand(fake_batch, fake_trglen, fake_trglen) > 0.5
        sample_src, sample_trg, sample_src_mask, sample_trg_mask = [t.to(self.device) for t in [sample_src, sample_trg, sample_src_mask, sample_trg_mask]]
        sample_encoded = self.encode(sample_src, sample_src_mask)
        sample_before_logits = self.decode(sample_trg, sample_encoded, sample_src_mask, sample_trg_mask)
        needed_fn = {'forward': (sample_src, sample_trg, sample_src_mask, sample_trg_mask), "encode": (sample_src, sample_src_mask), "decode": (sample_trg, sample_encoded, sample_src_mask, sample_trg_mask), "to_logits": sample_before_logits}
        traced_model = torch.jit.trace_module(self, needed_fn, check_trace=check_trace)
        torch.jit.save(traced_model, serve_path)
        return serve_path

    @property
    def fields(self):
        return (self.SRC, self.TRG)