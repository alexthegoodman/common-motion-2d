// use tokenizers::decoders::DecoderWrapper;
// use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
// use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence, NormalizerWrapper};
// use tokenizers::pre_tokenizers::byte_level::ByteLevel;
// use tokenizers::pre_tokenizers::PreTokenizerWrapper;
// use tokenizers::processors::PostProcessorWrapper;
// use tokenizers::{AddedToken, Model, Result, TokenizerBuilder};

// use std::path::Path;

use tokenizers::decoders::wordpiece::WordPiece as DecoderWordPiece;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::models::wordpiece::{WordPiece, WordPieceTrainerBuilder};
use tokenizers::normalizers::NFKC;
use tokenizers::pre_tokenizers::digits::Digits;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::tokenizer::Normalizer;
use tokenizers::{AddedToken, Error, TokenizerBuilder};

fn main() -> Result<(), Error> {
    // let mut trainer = BpeTrainerBuilder::new()
    //     .show_progress(true)
    //     .vocab_size(vocab_size)
    //     .min_frequency(0)
    //     .special_tokens(vec![
    //         AddedToken::from(String::from("<s>"), true),
    //         AddedToken::from(String::from("<pad>"), true),
    //         AddedToken::from(String::from("</s>"), true),
    //         AddedToken::from(String::from("<unk>"), true),
    //         AddedToken::from(String::from("<mask>"), true),
    //     ])
    //     .build();

    // let mut tokenizer = TokenizerBuilder::new()
    //     .with_model(BPE::default())
    //     .with_normalizer(Some(Sequence::new(vec![
    //         Strip::new(true, true).into(),
    //         NFC.into(),
    //     ])))
    //     .with_pre_tokenizer(Some(ByteLevel::default()))
    //     .with_post_processor(Some(ByteLevel::default()))
    //     .with_decoder(Some(ByteLevel::default()))
    //     .build()?;

    let vocab_size: usize = 12003;

    let mut trainer = WordPieceTrainerBuilder::new()
        .vocab_size(vocab_size) // Keep it small but enough for your patterns
        .special_tokens(vec![
            AddedToken::from(String::from("[CLS]"), true),
            AddedToken::from(String::from("[SEP]"), true),
            AddedToken::from(String::from("[PAD]"), true),
            AddedToken::from(String::from("[UNK]"), true),
            AddedToken::from(String::from("[NEG]"), true),
            AddedToken::from(String::from("[DEC]"), true),
        ])
        .build();

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(WordPiece::default())
        .with_normalizer(Some(NFKC))
        // Use Digits pre-tokenizer to handle numbers better
        .with_pre_tokenizer(Some(Digits::new(true)))
        .with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("[CLS] $A [SEP]")
                .unwrap()
                .try_pair("[CLS] $A [SEP] $B:1 [SEP]:1")
                .unwrap()
                .special_tokens(vec![("[CLS]", 1), ("[SEP]", 1), ("[NEG]", 1), ("[DEC]", 1)])
                .build()
                .unwrap(),
        ))
        .with_decoder(Some(DecoderWordPiece::default()))
        .build()?;

    let pretty = false;
    tokenizer
        .train_from_files(
            &mut trainer,
            vec!["D:/projects/common/common-motion-2d/backup/vocab-all.txt".to_string()],
        )?
        .save(
            "D:/projects/common/common-motion-2d/backup/tokenizer-all-wordpiece-ind.json",
            pretty,
        )?;

    Ok(())
}
