using Pkg
# Pkg.add(["Flux", "Random", "NNlib", "CUDA", "CSV", "DataFrames", "MLUtils", "Optimisers", "StatsBase", "OneHotArrays"])

using Flux, Random, NNlib, CSV, DataFrames, MLUtils, Optimisers, StatsBase, OneHotArrays
using Statistics: mean, std

# -----------------------------
# Model Hyperparameters
# -----------------------------
const VOCAB_SIZE = 100
const MAX_LEN = 8
const EMBED_DIM = 8
const FFN_DIM = 8
const NUM_CLASSES = 2

# -----------------------------
# Multi-Head Attention Layer
# -----------------------------
struct MultiHeadAttention
    embed_dim::Int
    num_heads::Int
    head_dim::Int
    wq::Dense
    wk::Dense
    wv::Dense
    wo::Dense
    
    function MultiHeadAttention(embed_dim::Int; num_heads::Int=1)
        @assert embed_dim % num_heads == 0
        head_dim = embed_dim ÷ num_heads
        
        wq = Dense(embed_dim => embed_dim, bias=false)
        wk = Dense(embed_dim => embed_dim, bias=false)
        wv = Dense(embed_dim => embed_dim, bias=false)
        wo = Dense(embed_dim => embed_dim, bias=false)
        
        new(embed_dim, num_heads, head_dim, wq, wk, wv, wo)
    end
    
    # Add explicit constructor for optimizer compatibility
    function MultiHeadAttention(embed_dim::Int, num_heads::Int, head_dim::Int, wq::Dense, wk::Dense, wv::Dense, wo::Dense)
        new(embed_dim, num_heads, head_dim, wq, wk, wv, wo)
    end
end

function (m::MultiHeadAttention)(x)
    # x shape: (embed_dim, seq_len, batch_size)
    embed_dim, seq_len, batch_size = size(x)
    
    # Linear projections - reshape for matrix multiplication
    x_flat = reshape(x, embed_dim, seq_len * batch_size)
    q = reshape(m.wq(x_flat), embed_dim, seq_len, batch_size)
    k = reshape(m.wk(x_flat), embed_dim, seq_len, batch_size)
    v = reshape(m.wv(x_flat), embed_dim, seq_len, batch_size)
    
    # Reshape for multi-head attention
    q = reshape(q, m.head_dim, m.num_heads, seq_len, batch_size)
    k = reshape(k, m.head_dim, m.num_heads, seq_len, batch_size)
    v = reshape(v, m.head_dim, m.num_heads, seq_len, batch_size)
    
    # Scaled dot-product attention
    scores = NNlib.batched_mul(
        permutedims(q, (1, 3, 2, 4)), 
        permutedims(k, (1, 3, 2, 4))
    ) ./ sqrt(Float32(m.head_dim))
    
    attn_weights = softmax(scores, dims=2)
    
    # Apply attention to values
    out = NNlib.batched_mul(attn_weights, permutedims(v, (1, 3, 2, 4)))
    
    # Concatenate heads and project
    out = reshape(out, embed_dim, seq_len, batch_size)
    out_flat = reshape(out, embed_dim, seq_len * batch_size)
    out = reshape(m.wo(out_flat), embed_dim, seq_len, batch_size)
    
    return out
end

Flux.@functor MultiHeadAttention

# -----------------------------
# Transformer Block
# -----------------------------
struct MiniTransformerBlock
    mha::MultiHeadAttention
    ffn::Chain
    norm1::LayerNorm
    norm2::LayerNorm
    
    # Add explicit constructor for optimizer compatibility
    function MiniTransformerBlock(mha::MultiHeadAttention, ffn::Chain, norm1::LayerNorm, norm2::LayerNorm)
        new(mha, ffn, norm1, norm2)
    end
end

function MiniTransformerBlock(embed_dim::Int, ffn_dim::Int)
    mha = MultiHeadAttention(embed_dim; num_heads=1)
    ffn = Chain(
        Dense(embed_dim => ffn_dim, relu),
        Dense(ffn_dim => embed_dim)
    )
    norm1 = LayerNorm(embed_dim)
    norm2 = LayerNorm(embed_dim)
    return MiniTransformerBlock(mha, ffn, norm1, norm2)
end

function (m::MiniTransformerBlock)(x)
    # Self-attention with residual connection and layer norm
    attn_out = m.mha(x)
    x_norm1 = m.norm1(x .+ attn_out)
    
    # Feed-forward with residual connection and layer norm
    # Reshape for FFN processing
    embed_dim, seq_len, batch_size = size(x_norm1)
    x_flat = reshape(x_norm1, embed_dim, seq_len * batch_size)
    ffn_out_flat = m.ffn(x_flat)
    ffn_out = reshape(ffn_out_flat, embed_dim, seq_len, batch_size)
    
    x_norm2 = m.norm2(x_norm1 .+ ffn_out)
    
    return x_norm2
end

Flux.@functor MiniTransformerBlock

# -----------------------------
# Positional Embedding Layer
# -----------------------------
struct AddPositionEmbedding
    pos_embed::AbstractArray{Float32,3}
    
    # Add explicit constructor for optimizer compatibility
    function AddPositionEmbedding(pos_embed::AbstractArray{Float32,3})
        new(pos_embed)
    end
end

function AddPositionEmbedding(max_len::Int, embed_dim::Int)
    pos_embed = randn(Float32, embed_dim, max_len, 1) .* 0.1f0
    return AddPositionEmbedding(pos_embed)
end

function (m::AddPositionEmbedding)(x)
    seq_len = size(x, 2)
    pos_embed = m.pos_embed[:, 1:seq_len, :]
    return x .+ pos_embed
end

Flux.@functor AddPositionEmbedding

# -----------------------------
# Global Average Pooling Layer
# -----------------------------
struct GlobalAvgPool end

function (::GlobalAvgPool)(x)
    # x shape: (embed_dim, seq_len, batch_size)
    # Output: (embed_dim, 1, batch_size)
    return mean(x, dims=2)
end

Flux.@functor GlobalAvgPool

# -----------------------------
# Reshape Layer
# -----------------------------
struct ReshapeLayer end

function (::ReshapeLayer)(x)
    # x shape: (embed_dim, 1, batch_size)
    # Output: (embed_dim, batch_size)
    return reshape(x, size(x, 1), size(x, 3))
end

Flux.@functor ReshapeLayer

# -----------------------------
# MiniBioBERT Model
# -----------------------------
function MiniBioBERT()
    return Chain(
        Embedding(VOCAB_SIZE => EMBED_DIM),                  # Token embedding
        AddPositionEmbedding(MAX_LEN, EMBED_DIM),           # Positional embedding
        MiniTransformerBlock(EMBED_DIM, FFN_DIM),           # Transformer block
        GlobalAvgPool(),                                     # Global average pooling
        ReshapeLayer(),                                      # Reshape for classification
        Dense(EMBED_DIM => NUM_CLASSES)                     # Classification head
    )
end

# -----------------------------
# Data Processing Functions
# -----------------------------
function tokenize(text)
    lowercase.(split(strip(text)))
end

function build_vocab(sentences, vocab_size=VOCAB_SIZE)
    tokens = reduce(vcat, map(tokenize, sentences))
    token_counts = StatsBase.countmap(tokens)
    sorted_tokens = sort(collect(token_counts), by=last, rev=true)
    
    vocab = Dict{String, Int}()
    vocab["<PAD>"] = 1  # Padding token
    vocab["<UNK>"] = 2  # Unknown token
    
    idx = 3
    for (token, count) in sorted_tokens
        if idx > vocab_size
            break
        end
        vocab[token] = idx
        idx += 1
    end
    
    return vocab
end

function encode(text, vocab, maxlen)
    tokens = tokenize(text)
    ids = [get(vocab, t, vocab["<UNK>"]) for t in tokens]
    
    # Pad or truncate to maxlen
    if length(ids) > maxlen
        ids = ids[1:maxlen]
    else
        ids = vcat(ids, fill(vocab["<PAD>"], maxlen - length(ids)))
    end
    
    return ids
end

function create_sample_data()
    # Enhanced biomedical sample data
    sentences = [
        "Aspirin relieves pain effectively",
        "Insulin controls blood sugar levels",
        "Headache may be severe symptom",
        "Cancer treatment varies by type",
        "Fever is common symptom",
        "Antibiotics fight bacterial infection",
        "Diabetes requires constant monitoring",
        "Surgery can be necessary treatment",
        "Medication has potential side effects",
        "Therapy helps patient recovery",
        "Chemotherapy treats cancer patients",
        "Symptoms include nausea fatigue",
        "Morphine reduces severe pain",
        "Pneumonia causes breathing difficulty",
        "Vaccination prevents disease spread",
        "Migraine causes head pain",
        "Antibiotics treat bacterial infections",
        "Chest pain indicates problems",
        "Radiation therapy kills cells",
        "Dizziness is common symptom"
    ]
    
    # 1 for treatment, 0 for symptom
    labels = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    return sentences, labels
end

function prepare_data(sentences, labels, vocab, maxlen=MAX_LEN)
    X = hcat([encode(sent, vocab, maxlen) for sent in sentences]...)
    Y = labels .+ 1  # Convert to 1-based indexing (1, 2)
    return X, Y
end

# -----------------------------
# Training Functions
# -----------------------------
function loss_fn(model, x, y)
    ŷ = model(x)
    # Apply softmax to ensure proper probabilities
    ŷ_softmax = softmax(ŷ, dims=1)
    loss = Flux.crossentropy(ŷ_softmax, OneHotArrays.onehotbatch(y, 1:NUM_CLASSES))
    return loss
end

function accuracy(ŷ, y)
    # Apply softmax for proper probability distribution
    ŷ_softmax = softmax(ŷ, dims=1)
    pred = OneHotArrays.onecold(ŷ_softmax, 1:NUM_CLASSES)
    return mean(pred .== y)
end

function train_model!(model, X_train, Y_train; epochs=200, lr=0.001, verbose=false)
    optimizer = Adam(lr)
    opt_state = Flux.setup(optimizer, model)
    n_samples = size(X_train, 2)
    
    for epoch in 1:epochs
        total_loss = 0.0
        total_acc = 0.0
        
        for i in 1:n_samples
            x = X_train[:, i:i]  # Single sample
            y = Y_train[i:i]     # Single label
            
            # Compute loss and gradients
            loss, grads = Flux.withgradient(model) do m
                loss_fn(m, x, y)
            end
            
            # Update parameters
            Flux.update!(opt_state, model, grads[1])
            
            # Track metrics
            ŷ = model(x)
            ŷ_softmax = softmax(ŷ, dims=1)
            total_loss += loss
            total_acc += accuracy(ŷ, y)
        end
        
        avg_loss = total_loss / n_samples
        avg_acc = total_acc / n_samples
        
        if verbose && epoch % 40 == 0
            println("  Epoch $epoch | Loss: $(round(avg_loss, digits=4)) | Acc: $(round(avg_acc, digits=4))")
        end
    end
    
    return model
end

function evaluate_model(model, X_test, Y_test)
    total_acc = 0.0
    n_samples = size(X_test, 2)
    
    for i in 1:n_samples
        x = X_test[:, i:i]
        y = Y_test[i:i]
        ŷ = model(x)
        acc = accuracy(ŷ, y)
        total_acc += acc
    end
    
    return total_acc / n_samples
end

# -----------------------------
# Cross-Validation Functions
# -----------------------------
function create_folds(n_samples::Int, k::Int=10)
    """Create k-fold cross-validation indices"""
    # Create shuffled indices
    indices = randperm(n_samples)
    
    # Calculate fold size
    fold_size = n_samples ÷ k
    remainder = n_samples % k
    
    folds = Vector{Vector{Int}}()
    
    start_idx = 1
    for i in 1:k
        # Add one extra sample to first 'remainder' folds
        current_fold_size = fold_size + (i <= remainder ? 1 : 0)
        end_idx = start_idx + current_fold_size - 1
        
        fold_indices = indices[start_idx:end_idx]
        push!(folds, fold_indices)
        
        start_idx = end_idx + 1
    end
    
    return folds
end

function k_fold_cross_validation(sentences, labels, vocab; k=10, epochs=200, lr=0.001)
    """Perform k-fold cross-validation"""
    println("Starting $k-fold cross-validation...")
    
    # Prepare full dataset
    X, Y = prepare_data(sentences, labels, vocab)
    n_samples = size(X, 2)
    
    # Create folds
    folds = create_folds(n_samples, k)
    
    # Store results
    fold_accuracies = Float64[]
    fold_details = []
    
    println("Data split into $k folds:")
    for i in 1:k
        println("  Fold $i: $(length(folds[i])) samples")
    end
    println()
    
    # Perform cross-validation
    for fold in 1:k
        println("Training fold $fold/$k...")
        
        # Split data
        test_indices = folds[fold]
        train_indices = setdiff(1:n_samples, test_indices)
        
        X_train = X[:, train_indices]
        Y_train = Y[train_indices]
        X_test = X[:, test_indices]
        Y_test = Y[test_indices]
        
        println("  Training samples: $(length(train_indices)), Test samples: $(length(test_indices))")
        
        # Create and train model
        model = MiniBioBERT()
        model = train_model!(model, X_train, Y_train, epochs=epochs, lr=lr, verbose=false)
        
        # Evaluate on test set
        test_acc = evaluate_model(model, X_test, Y_test)
        train_acc = evaluate_model(model, X_train, Y_train)
        
        push!(fold_accuracies, test_acc)
        push!(fold_details, (fold=fold, train_acc=train_acc, test_acc=test_acc, 
                           train_size=length(train_indices), test_size=length(test_indices)))
        
        println("  Fold $fold results: Train Acc: $(round(train_acc, digits=4)), Test Acc: $(round(test_acc, digits=4))")
        println()
    end
    
    return fold_accuracies, fold_details
end

function print_cv_results(fold_accuracies, fold_details)
    """Print detailed cross-validation results"""
    println("="^60)
    println("CROSS-VALIDATION RESULTS")
    println("="^60)
    
    println("Individual Fold Results:")
    println("-"^60)
    for detail in fold_details
        println("Fold $(detail.fold): Train=$(round(detail.train_acc, digits=4)), Test=$(round(detail.test_acc, digits=4)) ($(detail.train_size) train, $(detail.test_size) test)")
    end
    
    println("\nSummary Statistics:")
    println("-"^60)
    mean_acc = mean(fold_accuracies)
    std_acc = std(fold_accuracies)
    min_acc = minimum(fold_accuracies)
    max_acc = maximum(fold_accuracies)
    
    println("Mean Test Accuracy: $(round(mean_acc, digits=4)) ± $(round(std_acc, digits=4))")
    println("Min Test Accuracy:  $(round(min_acc, digits=4))")
    println("Max Test Accuracy:  $(round(max_acc, digits=4))")
    println("Range:              $(round(max_acc - min_acc, digits=4))")
    
    # Calculate confidence interval (assuming normal distribution)
    n = length(fold_accuracies)
    t_value = 2.262  # t-value for 95% CI with 9 degrees of freedom (10-fold CV)
    margin_of_error = t_value * std_acc / sqrt(n)
    ci_lower = mean_acc - margin_of_error
    ci_upper = mean_acc + margin_of_error
    
    println("95% Confidence Interval: [$(round(ci_lower, digits=4)), $(round(ci_upper, digits=4))]")
    println("="^60)
end

# -----------------------------
# Main Execution with Cross-Validation
# -----------------------------
function main()
    println("Initializing MiniBioBERT with 10-Fold Cross-Validation...")
    
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # Create sample data
    sentences, labels = create_sample_data()
    println("Sample sentences: $(length(sentences))")
    
    # Build vocabulary
    vocab = build_vocab(sentences)
    println("Vocabulary size: $(length(vocab))")
    
    # Count parameters (create temporary model)
    temp_model = MiniBioBERT()
    function count_params(model)
        total = 0
        for p in Flux.trainable(model)
            total += length(p)
        end
        return total
    end
    println("Total parameters: $(count_params(temp_model))")
    println()
    
    # Perform 10-fold cross-validation
    fold_accuracies, fold_details = k_fold_cross_validation(
        sentences, labels, vocab, 
        k=10, epochs=200, lr=0.001
    )
    
    # Print results
    print_cv_results(fold_accuracies, fold_details)
    
    # Train final model on full dataset for demonstration
    println("\nTraining final model on full dataset...")
    X, Y = prepare_data(sentences, labels, vocab)
    final_model = MiniBioBERT()
    final_model = train_model!(final_model, X, Y, epochs=200, lr=0.001, verbose=true)
    
    # Test with new sentences
    println("\nTesting final model with new sentences:")
    test_sentences = [
        "Paracetamol reduces fever",
        "Nausea is uncomfortable feeling",
        "Chemotherapy destroys cancer cells",
        "Vomiting is side effect",
        "Radiotherapy treats tumors"
    ]
    
    for sentence in test_sentences
        ids = encode(sentence, vocab, MAX_LEN)
        x = reshape(ids, :, 1)
        ŷ = final_model(x)
        ŷ_softmax = softmax(ŷ, dims=1)
        pred = OneHotArrays.onecold(ŷ_softmax, 1:NUM_CLASSES)[1]
        pred_name = pred == 1 ? "symptom" : "treatment"
        confidence = maximum(softmax(ŷ[:, 1]))
        println("'$sentence' → $pred_name (confidence: $(round(confidence, digits=3)))")
    end
    
    return final_model, vocab, fold_accuracies
end

# Run the main function
model, vocab, cv_results = main()