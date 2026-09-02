import Foundation

enum RiderSelectionState: Equatable, Sendable {
    case idle
    case selecting
    case selected
    case failed
    case lost
}

enum RiderMask: Equatable, Sendable {
    case normalizedPolygon([NormalizedVideoPoint])
}

enum RiderVideoSource: Hashable, Sendable {
    case asset(name: String)

    var identifier: String {
        switch self {
        case .asset(let name):
            "asset://\(name)"
        }
    }
}

struct RiderSelectionInput: Sendable {
    let videoSource: RiderVideoSource
    let timestamp: TimeInterval
    let frameIndex: Int?
    let point: NormalizedVideoPoint
}

struct RiderSelectionResult: Sendable {
    let selectionID: String
    let timestamp: TimeInterval
    let initialMask: RiderMask
    let confidence: Double?
    let session: any RiderTrackingSession
}

enum RiderSegmentationError: LocalizedError, Sendable {
    case riderNotFound
    case invalidInput

    var errorDescription: String? {
        switch self {
        case .riderNotFound:
            "Couldn't identify this rider"
        case .invalidInput:
            "The selected point is outside the video"
        }
    }
}

enum RiderSelectionTimeFormatter {
    static func string(from seconds: TimeInterval) -> String {
        let safeSeconds = max(0, Int(seconds.rounded(.down)))
        return String(format: "%02d:%02d", safeSeconds / 60, safeSeconds % 60)
    }
}

