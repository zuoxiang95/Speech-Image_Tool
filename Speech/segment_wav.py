import argparse
from collections import OrderedDict
import librosa
import numpy as np
import os


class Segment:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.next = None
        self.gap = 0

    def set_next(self, next):
        self.next = next
        self.gap = next.start - self.end

    def set_filename_and_id(self, filename, id):
        self.filename = filename
        self.id = id

    def merge_from(self, next):
        self.next = next.next
        self.gap = next.gap
        self.end = next.end

    def duration(self, sample_rate):
        return (self.end - self.start - 1) / sample_rate


def segment_wav(wav, threshold_db):
    # Find gaps at a fine resolution:
    parts = librosa.effects.split(wav, top_db=threshold_db, frame_length=1024, hop_length=256)

    # Build up a linked list of segments:
    head = None
    for start, end in parts:
        segment = Segment(start, end)
        if head is None:
            head = segment
        else:
            prev.set_next(segment)
        prev = segment
    return head


def find_best_merge(segments, sample_rate, max_duration, max_gap_duration):
    best = None
    best_score = 0
    s = segments
    while s.next is not None:
        gap_duration = s.gap / sample_rate
        merged_duration = (s.next.end - s.start) / sample_rate
        if gap_duration <= max_gap_duration and merged_duration <= max_duration:
            score = max_gap_duration - gap_duration
            if score > best_score:
                best = s
                best_score = score
        s = s.next
    return best


def find_segments(wav, sample_rate, min_duration, max_duration, max_gap_duration, threshold_db):
    segments = segment_wav(wav, threshold_db)

    # Merge until we can't merge any more
    while True:
        best = find_best_merge(segments, sample_rate, max_duration, max_gap_duration)
        if best is None:
            break
        best.merge_from(best.next)

    # Convert to list
    result = []
    s = segments
    while s is not None:
        # Exclude segments from the first 20 seconds -- these are the LibriVox disclaimer.
        # Also exclude segments that are too short or too long:
        if (s.end / sample_rate >= 20.0 and
                s.duration(sample_rate) >= min_duration and
                s.duration(sample_rate) <= max_duration):
            result.append(s)
        # Extend the end by 0.1 sec as we sometimes lose the ends of words ending in unvoiced sounds.
        s.end += int(0.1 * sample_rate)
        s = s.next

    # Exclude the last 2 segments: these are the end of section and reader ID.
    return result[:-2]


def load_filenames(source_name):
    mappings = OrderedDict()
    with open(os.path.join(os.path.dirname(__file__), 'sources', source_name + '.csv')) as f:
        for line in f:
            file_id, url = line.strip().split('|')
            mappings[file_id] = os.path.basename(url)
    return mappings


def build_segments(args):
    wav_dir = os.path.join(args.base_dir, 'wavs')
    os.makedirs(wav_dir, exist_ok=True)

    all_segments = []
    total_duration = 0
    filenames = load_filenames(args.source)
    for i, (file_id, filename) in enumerate(filenames.items()):
        print('Loading %s: %s (%d of %d)' % (file_id, filename, i + 1, len(filenames)))
        wav, sample_rate = librosa.load(os.path.join(args.base_dir, 'inputs', filename))
        print(' -> Loaded %.1f min of audio. Splitting...' % (len(wav) / sample_rate / 60))

        # Find segments
        segments = find_segments(wav, sample_rate, args.min_duration, args.max_duration,
                                 args.max_gap_duration, args.threshold)
        duration = sum((s.duration(sample_rate) for s in segments))
        total_duration += duration

        # Create records for the segments
        for j, s in enumerate(segments):
            all_segments.append(s)
            s.set_filename_and_id(filename, '%s-%04d' % (file_id, j + 1))
        print(' -> Segmented into %d parts (%.1f min, %.2f sec avg)' % (
            len(segments), duration / 60, duration / len(segments)))

        # Write segments to disk:
        for s in segments:
            segment_wav = (wav[s.start:s.end] * 32767).astype(np.int16)
            out_path = os.path.join(wav_dir, '%s.wav' % s.id)
            librosa.output.write_wav(out_path, segment_wav, sample_rate)
            duration += len(segment_wav) / sample_rate
        print(' -> Wrote %d segment wav files' % len(segments))
        print(' -> Progress: %d segments, %.2f hours, %.2f sec avg' % (
            len(all_segments), total_duration / 3600, total_duration / len(all_segments)))

    print('Writing metadata for %d segments (%.2f hours)' % (len(all_segments), total_duration / 3600))
    with open(os.path.join(args.base_dir, 'segments.csv'), 'w') as f:
        for s in all_segments:
            f.write('%s|%s|%d|%d\n' % (s.id, s.filename, s.start, s.end))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='/experiments/audio')
    parser.add_argument('--source', default='LJ', help='Name of the source to process')
    parser.add_argument('--min_duration', type=float, default=1.0, help='In seconds')
    parser.add_argument('--max_duration', type=float, default=10.0, help='In seconds')
    parser.add_argument('--max_gap_duration', type=float, default=0.75, help='In seconds')
    parser.add_argument('--threshold', type=float, default=40.0, help='In decibels below max')
    args = parser.parse_args()
    build_segments(args)


if __name__ == "__main__":
    main()
