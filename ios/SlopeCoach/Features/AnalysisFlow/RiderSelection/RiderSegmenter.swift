import Foundation

protocol RiderTrackingSession: Sendable {
    var selectionID: String { get }
    var anchorTimestamp: TimeInterval { get }

    func mask(at timestamp: TimeInterval) async -> RiderMask?
}

protocol RiderSegmenter: Sendable {
    func selectRider(input: RiderSelectionInput) async throws -> RiderSelectionResult
}

