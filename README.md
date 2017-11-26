## kbecomp

Knowledge Base Embeddings Comparison

### Lofty Vision

I'd love to implement "all" (many) of the knowledge base completion methods
in the literature. These largely stem from (cite) the TransE model of
[Translating Embeddings for Modeling Multi-relational Data](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf),
and [SME](https://github.com/glorotxa/SME).
I'd love, also, to have all the standard datasets in one place.
And be able to actually write down and see all the options for a model,
and run all the combinations, easily. Here, I'm thinking not just of things
like the embedding dimension, but various training strategies (negative sampling,
optimizers).

Once I've got that setup, I'd like to run a bunch of models over and over,
and instead of producing "the score" of a model on a dataset, be able to
provide "the distribution of scores" of a model on a dataset (understanding,
again, all the choices in the implementation of it).

Its very possible I should just contribute to
[KB2E](https://github.com/thunlp/KB2E) (though that's c++, not pytorch).


### Current Reality

I've never written anything in pytorch before, but wanted to give it a try.
I've started with TransE, because that's the model I know the best, and is
a standard baseline in the literature anyway. It's nearly there, but still
requires better evaluation/validation metrics (e.g., HITS@k). It'll then
require some generalizations (e.g., no hard-coded paths). But it's a start.
