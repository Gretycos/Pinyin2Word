import torch.nn as nn

class ModelEmbeddings(nn.Module):
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layers.

        @param embed_size (int): Embedding size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        """
        super(ModelEmbeddings,self).__init__()
        self.embed_size = embed_size

        # 默认值
        self.source = None
        self.target = None

        src_pad_token_idx = vocab.src['<pad>']
        tgt_pad_token_idx = vocab.tgt['<pad>']

        # 初始化embedding层
        self.source = nn.Embedding(len(vocab.src),self.embed_size,src_pad_token_idx)
        self.target = nn.Embedding(len(vocab.tgt),self.embed_size,tgt_pad_token_idx)
