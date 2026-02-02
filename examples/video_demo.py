"""
Exemple d'utilisation du processeur vidéo DINOv3
Compatible Mac M3 Max (MPS)
"""

import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from video_processor import VideoProcessor, VideoConfig, visualize_video_features


def demo_basic():
    """Exemple basique: extraire les features d'une vidéo."""
    # Configuration adaptée au Mac M3 Max
    config = VideoConfig(
        batch_size=8,      # M3 Max peut gérer des batches plus grands
        frame_skip=1,      # Toutes les frames
        max_frames=300,    # Limiter pour le test
    )

    # Initialiser le processeur
    processor = VideoProcessor(config=config)

    # Chemin vers ta vidéo
    video_path = Path(__file__).parent.parent / "data" / "test_video.mp4"

    if not video_path.exists():
        print(f"Video not found: {video_path}")
        print("Please provide a video file.")
        return

    # Traiter la vidéo
    results = processor.process_video(str(video_path))

    # Afficher les résultats
    print(f"\n=== Results ===")
    print(f"Frames processed: {results['num_frames']}")
    print(f"Processing speed: {results['num_frames']/results['processing_time']:.1f} FPS")
    print(f"CLS token dimension: {results['cls_tokens'].shape[1]}")

    # Trouver les keyframes (changements de scène)
    keyframes = processor.find_keyframes(results['cls_tokens'], threshold=0.85)
    print(f"Scene changes detected: {len(keyframes) - 1}")

    # Visualiser
    visualize_video_features(results)


def demo_webcam():
    """Exemple avec webcam en temps réel."""
    import cv2
    import torch
    from torchvision import transforms

    config = VideoConfig(batch_size=1)
    processor = VideoProcessor(config=config)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit")

    prev_cls = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = transform(frame_rgb).unsqueeze(0).to(processor.device)

        # Extract features
        with torch.no_grad():
            features = processor.model.get_intermediate_layers(img_tensor, n=1)[0]
            cls_token = features[:, 0]
            cls_token = torch.nn.functional.normalize(cls_token, dim=-1)

        # Compute similarity with previous frame
        if prev_cls is not None:
            similarity = torch.nn.functional.cosine_similarity(
                cls_token, prev_cls, dim=1
            ).item()
            cv2.putText(frame, f"Similarity: {similarity:.3f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        prev_cls = cls_token

        cv2.imshow('DINOv3 Real-time', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def demo_similarity_search():
    """Exemple: trouver des frames similaires dans une vidéo."""
    import torch

    config = VideoConfig(
        batch_size=8,
        frame_skip=5,  # Échantillonner pour accélérer
    )

    processor = VideoProcessor(config=config)

    video_path = Path(__file__).parent.parent / "data" / "test_video.mp4"
    results = processor.process_video(str(video_path))

    cls_tokens = results['cls_tokens']

    # Prendre la première frame comme requête
    query = cls_tokens[0:1]

    # Calculer les similarités
    similarities = torch.nn.functional.cosine_similarity(
        query.expand(cls_tokens.shape[0], -1),
        cls_tokens,
        dim=1
    )

    # Top-5 frames les plus similaires
    top_k = 5
    top_indices = similarities.argsort(descending=True)[:top_k]

    print(f"\nTop {top_k} frames most similar to frame 0:")
    for i, idx in enumerate(top_indices):
        frame_num = results['frame_indices'][idx]
        sim = similarities[idx].item()
        print(f"  {i+1}. Frame {frame_num}: similarity = {sim:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['basic', 'webcam', 'search'],
                       default='basic', help='Demo mode')
    args = parser.parse_args()

    if args.mode == 'basic':
        demo_basic()
    elif args.mode == 'webcam':
        demo_webcam()
    elif args.mode == 'search':
        demo_similarity_search()
