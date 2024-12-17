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

pub struct Gpt2Tokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl Default for Gpt2Tokenizer {
    fn default() -> Self {
        let mut tokenizer = tokenizers::Tokenizer::from_pretrained("gpt2", None).unwrap();
        tokenizer.add_special_tokens(&[
            tokenizers::AddedToken::from("[START]", true),
            tokenizers::AddedToken::from("[END]", true),
            tokenizers::AddedToken::from("[PAD]", true),
        ]);

        Self { tokenizer }
    }
}

impl Tokenizer for Gpt2Tokenizer {
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

// TODO: consider much more lightweight tokenizer
// use tokenizers::pre_tokenizers::split::Split;
// use tokenizers::tokenizer::Tokenizer;

// pub struct CustomTokenizer {
//     pub split: Split,
// }

// impl Tokenizer for CustomTokenizer {
//     fn tokenize(&self, input: &str) -> Vec<String> {
//         self.split.tokenize(input)
//     }
// }

// fn main() {
//     let tokenizer = CustomTokenizer {
//         split: Split::new(","),
//     };

//     let input = "1,2,3,4,5";
//     let tokens = tokenizer.tokenize(input);
//     println!("{:?}", tokens); // Output: ["1", "2", "3", "4", "5"]
// }
