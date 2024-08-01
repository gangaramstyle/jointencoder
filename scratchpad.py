import marimo

__generated_with = "0.7.12"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import torch
    from torch.nn import Module
    from torch import nn
    from einops import rearrange, repeat, reduce, pack, unpack

    def divisible_by(numer, denom):
        return (numer % denom) == 0

    def exists(val):
        return val is not None

    def at_most_one_of(*bools):
        return sum(map(int, bools)) <= 1

    class LayerNorm(Module):
        def __init__(
            self,
            dim,
            unit_offset = False
        ):
            """
            bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
            """
            super().__init__()
            self.unit_offset = unit_offset

            self.ln = nn.LayerNorm(dim, elementwise_affine = False)
            self.gamma = nn.Parameter(torch.ones(dim))
            nn.init.constant_(self.gamma, 1. - float(unit_offset))

        def forward(self, x):
            normed = self.ln(x)
            gamma = self.gamma + float(self.unit_offset)
            return normed * gamma


    class ViTransformerWrapper(Module):
        def __init__(
            self,
            *,
            image_size,
            patch_size,
            attn_layers,
            channels = 3,
            num_classes = None,
            post_emb_norm = False,
            num_register_tokens = 0,
            emb_dropout = 0.
        ):
            super().__init__()
            assert divisible_by(image_size, patch_size), 'image dimensions must be divisible by the patch size'
            dim = attn_layers.dim
            num_patches = (image_size // patch_size) ** 2
            patch_dim = channels * patch_size ** 2

            self.patch_size = patch_size

            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

            has_register_tokens = num_register_tokens > 0
            self.has_register_tokens = has_register_tokens

            if has_register_tokens:
                self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

            self.patch_to_embedding = nn.Sequential(
                LayerNorm(patch_dim),
                nn.Linear(patch_dim, dim),
                LayerNorm(dim)
            )

            self.post_emb_norm = LayerNorm(dim) if post_emb_norm else nn.Identity()
            self.dropout = nn.Dropout(emb_dropout)

            self.attn_layers = attn_layers

            self.mlp_head = nn.Linear(dim, num_classes) if exists(num_classes) else nn.Identity()

        def forward(
            self,
            img,
            return_embeddings = False,
            return_logits_and_embeddings = False
        ):
            b, p = img.shape[0], self.patch_size

            x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
            x = self.patch_to_embedding(x)
            n = x.shape[1]

            x = x + self.pos_embedding[:, :n]

            x = self.post_emb_norm(x)
            x = self.dropout(x)

            if self.has_register_tokens:
                r = repeat(self.register_tokens, 'n d -> b n d', b = b)
                x, ps = pack((x, r), 'b * d')

            embed = self.attn_layers(x)

            if self.has_register_tokens:
                embed, _ = unpack(embed, ps, 'b * d')

            assert at_most_one_of(return_embeddings, return_logits_and_embeddings)

            if not exists(self.mlp_head) or return_embeddings:
                return embed

            pooled = embed.mean(dim = -2)
            logits = self.mlp_head(pooled)

            if not return_logits_and_embeddings:
                return logits

            return logits, embed
    return (
        LayerNorm,
        Module,
        ViTransformerWrapper,
        at_most_one_of,
        divisible_by,
        exists,
        mo,
        nn,
        pack,
        rearrange,
        reduce,
        repeat,
        torch,
        unpack,
    )


@app.cell
def __(ViTransformerWrapper, torch):
    from x_transformers import Encoder

    encoder = ViTransformerWrapper(
        image_size = 256,
        patch_size = 32,
        attn_layers = Encoder(
            dim = 256,
            depth = 6,
            heads = 8
        ),
        emb_dropout = 0.5
    )


    img = torch.randn(1, 3, 256, 256)

    encoded = encoder(img, return_embeddings = True)

    print(encoded.shape)
    return Encoder, encoded, encoder, img


@app.cell
def __():
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
