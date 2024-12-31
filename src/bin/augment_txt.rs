use rand::Rng;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

#[derive(Debug, Clone)]
struct DataPoint {
    polygon_index: String,
    time: f64,
    width: f64,
    height: f64,
    x: f64,
    y: f64,
}

impl DataPoint {
    fn from_line(line: &str) -> Option<Self> {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 6 {
            return None;
        }

        Some(DataPoint {
            polygon_index: parts[0].trim().to_string(),
            time: parts[1].trim().parse().unwrap_or(0.0),
            width: parts[2].trim().parse().unwrap_or(0.0),
            height: parts[3].trim().parse().unwrap_or(0.0),
            x: parts[4].trim().parse().unwrap_or(0.0),
            y: parts[5].trim().parse().unwrap_or(0.0),
        })
    }

    fn to_string(&self) -> String {
        format!(
            "{}, {}, {}, {}, {}, {}",
            self.polygon_index,
            self.time,
            self.width.round(),
            self.height.round(),
            self.x.round(),
            self.y.round()
        )
    }
}

fn main() -> std::io::Result<()> {
    let input_path = "D:/projects/common/common-motion-2d/backup/train.txt";
    let output_path = "D:/projects/common/common-motion-2d/backup/augmented.txt";
    let num_augmentations = 5;

    let file = File::open(input_path)?;
    let reader = BufReader::new(file);
    let mut output = File::create(output_path)?;

    let mut current_sequence = Vec::new();
    let mut is_reading_sequence = false;

    for line in reader.lines() {
        let line = line?;

        match line.as_str() {
            "---" => {
                if !current_sequence.is_empty() {
                    // Write original sequence
                    for line in &current_sequence {
                        writeln!(output, "{}", line)?;
                    }
                    writeln!(output, "---")?;

                    // Create augmented copies
                    for _ in 0..num_augmentations {
                        let augmented = augment_sequence(&current_sequence);
                        for line in augmented {
                            writeln!(output, "{}", line)?;
                        }
                        writeln!(output, "---")?;
                    }
                }
                current_sequence.clear();
                is_reading_sequence = false;
            }
            "" => continue,
            _ => {
                is_reading_sequence = true;
                current_sequence.push(line);
            }
        }
    }

    Ok(())
}

fn augment_sequence(sequence: &[String]) -> Vec<String> {
    let mut rng = rand::thread_rng();
    let mut augmented = Vec::new();
    let mut saw_separator = false;

    // Store prompt data for correlation
    let mut prompt_data: HashMap<String, Vec<DataPoint>> = HashMap::new();
    let mut completion_points: Vec<DataPoint> = Vec::new();

    // First pass: collect prompt data and parse completion points
    for line in sequence {
        if line == "!!!" {
            saw_separator = true;
            augmented.push(line.clone());
            continue;
        }

        if let Some(point) = DataPoint::from_line(line) {
            if !saw_separator {
                prompt_data
                    .entry(point.polygon_index.clone())
                    .or_insert_with(Vec::new)
                    .push(point);
            } else {
                completion_points.push(point);
            }
        }
    }

    // Second pass: generate augmented data
    saw_separator = false;
    for line in sequence {
        if line == "!!!" {
            saw_separator = true;
            augmented.push(line.clone());
            continue;
        }

        if let Some(mut point) = DataPoint::from_line(line) {
            if !saw_separator {
                // Augment prompt portion
                let width_variation = rng.gen_range(0.95..1.05);
                let height_variation = rng.gen_range(0.95..1.05);
                point.width *= width_variation;
                point.height *= height_variation;
                point.x += rng.gen_range(-10.0..10.0);
                point.y += rng.gen_range(-10.0..10.0);

                // Store augmented values for completion correlation
                prompt_data
                    .entry(point.polygon_index.clone())
                    .or_insert_with(Vec::new)
                    .push(point.clone());
            } else {
                // Augment completion portion
                if let Some(prompt_points) = prompt_data.get(&point.polygon_index) {
                    // Match width and height from prompt
                    for prompt_point in prompt_points {
                        // if prompt_point.time == point.time {
                        // println!("Matched time: {}", point.time);
                        point.width = prompt_point.width;
                        point.height = prompt_point.height;

                        // Match x/y coordinates for time values 5 and 15.0
                        if (point.time - 5.0).abs() < 0.1 || (point.time - 15.0).abs() < 0.1 {
                            for p in prompt_points {
                                // if (p.time - point.time).abs() < 0.1 {
                                // println!("Matched x/y: {}", point.time);
                                point.x = p.x;
                                point.y = p.y;
                                // break;
                                // }
                            }
                        }
                        // break;
                        // }
                    }
                }
            }
            augmented.push(point.to_string());
        } else {
            augmented.push(line.clone());
        }
    }

    augmented
}
