use std::borrow::Borrow;
use std::ops::Deref;

use tokenizers::Encoding;

#[allow(dead_code)]
pub trait Tokenizer: Send + Sync {
    fn encode(&self, value: &str, special_tokens: bool) -> Vec<usize>;
    fn decode(&self, tokens: &[usize]) -> String;
    fn vocab_size(&self) -> usize;
    fn pad_token(&self) -> usize;
    fn start_token(&self) -> usize;
    fn end_token(&self) -> usize;
    fn pad_token_value(&self) -> String {
        self.decode(&[self.pad_token()])
    }
    fn start_token_value(&self) -> String {
        self.decode(&[self.start_token()])
    }
    fn end_token_value(&self) -> String {
        self.decode(&[self.end_token()])
    }
}

pub struct NumericalTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl Default for NumericalTokenizer {
    fn default() -> Self {
        // let mut tokenizer = tokenizers::Tokenizer::from_pretrained("gpt2", None).unwrap();
        let mut tokenizer = tokenizers::Tokenizer::from_file(
            "D:/projects/common/common-motion-2d/backup/tokenizer.json",
        )
        .unwrap();
        tokenizer.add_special_tokens(&[
            tokenizers::AddedToken::from("[START]", true),
            tokenizers::AddedToken::from("[END]", true),
            tokenizers::AddedToken::from("[PAD]", true),
            tokenizers::AddedToken::from(String::from("<s>"), true),
            tokenizers::AddedToken::from(String::from("<pad>"), true),
            tokenizers::AddedToken::from(String::from("</s>"), true),
            tokenizers::AddedToken::from(String::from("<unk>"), true),
            tokenizers::AddedToken::from(String::from("<mask>"), true),
        ]);

        Self { tokenizer }
    }
}

impl Tokenizer for NumericalTokenizer {
    fn encode(&self, value: &str, special_tokens: bool) -> Vec<usize> {
        let text = match special_tokens {
            true => "[START]".to_owned() + value + "[END]",
            false => value.to_string(),
        };
        let tokens = self.tokenizer.encode(text, true).unwrap();
        tokens.get_ids().iter().map(|t| *t as usize).collect()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        let tokens = tokens.iter().map(|t| *t as u32).collect::<Vec<u32>>();
        self.tokenizer.decode(&tokens, false).unwrap()
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    fn pad_token(&self) -> usize {
        self.tokenizer.token_to_id("[PAD]").unwrap() as usize
    }

    fn start_token(&self) -> usize {
        self.tokenizer.token_to_id("[START]").unwrap() as usize
    }

    fn end_token(&self) -> usize {
        self.tokenizer.token_to_id("[END]").unwrap() as usize
    }
}

impl NumericalTokenizer {
    pub fn encode_inference(&self, value: &str, special_tokens: bool) -> Encoding {
        let text = match special_tokens {
            true => "[START]".to_owned() + value + "[END]",
            false => value.to_string(),
        };
        let tokens = self.tokenizer.encode(text, true).unwrap();
        tokens
    }

    pub fn decode_inference(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(&tokens, false).unwrap()
    }

    pub fn pad(&self, tokens: Vec<usize>, max_length: usize) -> Vec<usize> {
        let pad_token = self.pad_token();
        let mut padded_tokens = tokens;
        while padded_tokens.len() < max_length {
            padded_tokens.push(pad_token);
        }
        padded_tokens
    }
}
