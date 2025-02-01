use burn::data::dataset::{Dataset, InMemDataset};
use derive_new::new;
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::copy,
    path::{Path, PathBuf},
};

use std::io::{self, BufRead, BufReader};

pub struct MotionDataset {
    sequences: InMemDataset<String>, // Store entire sequences instead of prompts/completions
}

#[derive(new, Clone, Debug)]
pub struct TextGenerationItem {
    pub sequence: String, // Single sequence for training
}

impl MotionDataset {
    pub fn new() -> Result<Self, io::Error> {
        Self::train()
    }

    pub fn train() -> Result<Self, io::Error> {
        let path = MotionDataset::download_train();
        let sequences = Self::load_txt_dataset(&path)?;
        Ok(Self { sequences })
    }

    pub fn test() -> Result<Self, io::Error> {
        let path = MotionDataset::download_test();
        let sequences = Self::load_txt_dataset(&path)?;
        Ok(Self { sequences })
    }

    fn load_txt_dataset(path: &Path) -> Result<InMemDataset<String>, io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut sequences = Vec::new();

        let mut current_sequence = String::new();

        for line in reader.lines() {
            let line = line?;

            if line.trim() == "---" {
                if !current_sequence.is_empty() {
                    sequences.push(current_sequence.trim().to_string());
                    current_sequence.clear();
                }
            } else if !line.trim().is_empty() {
                current_sequence.push_str(&line);
                current_sequence.push(','); // comma delimiter to separate lines clearly
                current_sequence.push(' '); // space for consistency and separation
                current_sequence.push('\n'); // Add newline for separation
            }
        }

        // Handle the last sequence if it exists
        if !current_sequence.is_empty() {
            sequences.push(current_sequence.trim().to_string());
        }

        Ok(InMemDataset::new(sequences))
    }

    fn download_train() -> PathBuf {
        let backup_dir = Path::new("backup");
        let file_name = backup_dir.join("augmented.txt");

        if file_name.exists() {
            println!("File already downloaded at {:?}", file_name);
        }

        file_name
    }

    fn download_test() -> PathBuf {
        let backup_dir = Path::new("backup");
        let file_name = backup_dir.join("test.txt");

        if file_name.exists() {
            println!("File already downloaded at {:?}", file_name);
        }

        file_name
    }
}

impl Dataset<TextGenerationItem> for MotionDataset {
    fn get(&self, index: usize) -> Option<TextGenerationItem> {
        self.sequences
            .get(index)
            .map(|sequence| TextGenerationItem::new(sequence))
    }

    fn len(&self) -> usize {
        self.sequences.len()
    }
}

// pub struct MotionDataset {
//     prompts: InMemDataset<String>,
//     completions: InMemDataset<String>,
// }

// #[derive(new, Clone, Debug)]
// pub struct TextGenerationItem {
//     pub prompt: String,
//     pub completion: String,
// }

// impl MotionDataset {
//     pub fn new() -> Result<Self, io::Error> {
//         Self::train()
//     }

//     pub fn train() -> Result<Self, io::Error> {
//         let path = MotionDataset::download_train();
//         let (prompts, completions) = Self::load_txt_dataset(&path)?;
//         Ok(Self {
//             prompts,
//             completions,
//         })
//     }

//     pub fn test() -> Result<Self, io::Error> {
//         let path = MotionDataset::download_test();
//         let (prompts, completions) = Self::load_txt_dataset(&path)?;
//         Ok(Self {
//             prompts,
//             completions,
//         })
//     }

//     fn load_txt_dataset(
//         path: &Path,
//     ) -> Result<(InMemDataset<String>, InMemDataset<String>), io::Error> {
//         let file = File::open(path)?;
//         let reader = BufReader::new(file);
//         let mut prompts = Vec::new();
//         let mut completions = Vec::new();

//         let mut current_sequence = String::new();

//         for line in reader.lines() {
//             let line = line?;

//             if line.trim() == "---" {
//                 if !current_sequence.is_empty() {
//                     // Split sequence at !!! separator
//                     if let Some((prompt, completion)) = current_sequence.split_once("!!!") {
//                         prompts.push(prompt.trim().to_string());
//                         completions.push(completion.trim().to_string());
//                     }
//                     current_sequence.clear();
//                 }
//             } else if !line.trim().is_empty() {
//                 current_sequence.push_str(&line);
//                 current_sequence.push(','); // comma delimiter to separate lines clearly
//                 current_sequence.push(' '); // space for consistency and separation
//                 current_sequence.push('\n');
//             }
//         }

//         // Handle the last sequence if it exists
//         if !current_sequence.is_empty() {
//             if let Some((prompt, completion)) = current_sequence.split_once("!!!") {
//                 prompts.push(prompt.trim().to_string());
//                 completions.push(completion.trim().to_string());
//             }
//         }

//         Ok((InMemDataset::new(prompts), InMemDataset::new(completions)))
//     }

//     fn download_train() -> PathBuf {
//         let backup_dir = Path::new("backup");
//         let file_name = backup_dir.join("augmented.txt");

//         if file_name.exists() {
//             println!("File already downloaded at {:?}", file_name);
//         }

//         file_name
//     }

//     fn download_test() -> PathBuf {
//         let backup_dir = Path::new("backup");
//         let file_name = backup_dir.join("test.txt");

//         if file_name.exists() {
//             println!("File already downloaded at {:?}", file_name);
//         }

//         file_name
//     }
// }

// impl Dataset<TextGenerationItem> for MotionDataset {
//     fn get(&self, index: usize) -> Option<TextGenerationItem> {
//         match (self.prompts.get(index), self.completions.get(index)) {
//             (Some(prompt), Some(completion)) => Some(TextGenerationItem::new(prompt, completion)),
//             _ => None,
//         }
//     }

//     fn len(&self) -> usize {
//         self.prompts.len()
//     }
// }
