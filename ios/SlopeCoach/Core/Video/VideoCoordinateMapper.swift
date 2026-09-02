import CoreGraphics

struct NormalizedVideoPoint: Equatable, Hashable, Sendable {
    let x: CGFloat
    let y: CGFloat

    init?(x: CGFloat, y: CGFloat) {
        guard x.isFinite, y.isFinite, (0...1).contains(x), (0...1).contains(y) else {
            return nil
        }

        self.x = x
        self.y = y
    }
}

enum VideoContentMode: Sendable {
    case aspectFit
    case aspectFill
}

struct VideoCoordinateMapper: Sendable {
    let sourceSize: CGSize
    let containerSize: CGSize
    let contentMode: VideoContentMode

    var displayedVideoRect: CGRect {
        guard sourceSize.width > 0,
              sourceSize.height > 0,
              containerSize.width > 0,
              containerSize.height > 0 else {
            return .zero
        }

        let widthScale = containerSize.width / sourceSize.width
        let heightScale = containerSize.height / sourceSize.height
        let scale = contentMode == .aspectFit
            ? min(widthScale, heightScale)
            : max(widthScale, heightScale)
        let displayedSize = CGSize(
            width: sourceSize.width * scale,
            height: sourceSize.height * scale
        )

        return CGRect(
            x: (containerSize.width - displayedSize.width) / 2,
            y: (containerSize.height - displayedSize.height) / 2,
            width: displayedSize.width,
            height: displayedSize.height
        )
    }

    func normalizedPoint(forDisplayPoint point: CGPoint) -> NormalizedVideoPoint? {
        let videoRect = displayedVideoRect
        guard videoRect.width > 0,
              videoRect.height > 0,
              point.x >= videoRect.minX,
              point.x <= videoRect.maxX,
              point.y >= videoRect.minY,
              point.y <= videoRect.maxY else {
            return nil
        }

        return NormalizedVideoPoint(
            x: min(max((point.x - videoRect.minX) / videoRect.width, 0), 1),
            y: min(max((point.y - videoRect.minY) / videoRect.height, 0), 1)
        )
    }

    func displayPoint(for normalizedPoint: NormalizedVideoPoint) -> CGPoint {
        let videoRect = displayedVideoRect
        return CGPoint(
            x: videoRect.minX + normalizedPoint.x * videoRect.width,
            y: videoRect.minY + normalizedPoint.y * videoRect.height
        )
    }

    func displayPoints(for mask: RiderMask) -> [CGPoint] {
        switch mask {
        case .normalizedPolygon(let points):
            points.map(displayPoint(for:))
        }
    }
}

